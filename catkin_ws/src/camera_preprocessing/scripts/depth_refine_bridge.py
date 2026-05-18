#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CameraInfo, Image


def image_msg_to_numpy(msg):
    encoding = msg.encoding
    if encoding == "16UC1":
        dtype = np.uint16
        channels = 1
    elif encoding == "32FC1":
        dtype = np.float32
        channels = 1
    else:
        raise ValueError("Unsupported image encoding: {}".format(encoding))

    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    itemsize = dtype.itemsize
    row_items = msg.step // itemsize
    data = np.frombuffer(msg.data, dtype=dtype)
    data = data.reshape((msg.height, row_items))
    image = data[:, : msg.width * channels]
    if channels > 1:
        image = image.reshape((msg.height, msg.width, channels))
    else:
        image = image.reshape((msg.height, msg.width))
    return np.ascontiguousarray(image)


def numpy_to_image_msg(image, encoding, header):
    out = Image()
    out.header = header
    out.height = int(image.shape[0])
    out.width = int(image.shape[1])
    out.encoding = encoding
    out.is_bigendian = 0
    out.step = int(image.strides[0])
    out.data = np.ascontiguousarray(image).tobytes()
    return out


class DepthRefineBridge:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/d435/depth/image_rect_raw")
        self.output_topic = rospy.get_param("~output_topic", "/d435/depth/image_rect_refined")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/d435/depth/camera_info")
        self.use_camera_info = bool(rospy.get_param("~use_camera_info", True))
        self.preserve_input_header_stamp = bool(
            rospy.get_param("~preserve_input_header_stamp", True)
        )

        # Typical D435 depth in 16UC1 is millimeters. Convert to meters internally.
        self.depth_scale = float(rospy.get_param("~depth_scale", 0.001))
        self.min_depth_m = float(rospy.get_param("~min_depth_m", 0.15))
        self.max_depth_m = float(rospy.get_param("~max_depth_m", 8.0))

        # Low-latency defaults: keep only lightweight filtering by default.
        self.enable_median = bool(rospy.get_param("~enable_median", True))
        self.median_ksize = int(rospy.get_param("~median_ksize", 3))
        if self.median_ksize % 2 == 0:
            self.median_ksize += 1

        # Bilateral is edge-preserving but expensive; disabled by default for realtime.
        self.enable_bilateral = bool(rospy.get_param("~enable_bilateral", False))
        self.bilateral_d = int(rospy.get_param("~bilateral_d", 5))
        self.bilateral_sigma_color = float(rospy.get_param("~bilateral_sigma_color", 0.06))
        self.bilateral_sigma_space = float(rospy.get_param("~bilateral_sigma_space", 5.0))

        self.enable_hole_filling = bool(rospy.get_param("~enable_hole_filling", True))
        self.hole_fill_max_iters = int(rospy.get_param("~hole_fill_max_iters", 2))
        self.hole_fill_kernel = int(rospy.get_param("~hole_fill_kernel", 3))
        if self.hole_fill_kernel < 1:
            self.hole_fill_kernel = 1
        if self.hole_fill_kernel % 2 == 0:
            self.hole_fill_kernel += 1
        self.hole_fill_kernel_mat = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.hole_fill_kernel, self.hole_fill_kernel),
        )

        self.enable_temporal_ema = bool(rospy.get_param("~enable_temporal_ema", False))
        self.temporal_alpha = float(rospy.get_param("~temporal_alpha", 0.4))
        self.prev_filtered_m = None

        self.latest_camera_info = None

        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=1, tcp_nodelay=True)
        self.sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.on_depth,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.camera_info_sub = None
        if self.use_camera_info:
            self.camera_info_sub = rospy.Subscriber(
                self.camera_info_topic,
                CameraInfo,
                self.on_camera_info,
                queue_size=1,
                tcp_nodelay=True,
            )

        rospy.loginfo(
            "Depth refine bridge ready: %s -> %s (camera_info=%s, keep_stamp=%s, low_latency=true)",
            self.input_topic,
            self.output_topic,
            self.use_camera_info,
            self.preserve_input_header_stamp,
        )

    def on_camera_info(self, msg):
        self.latest_camera_info = msg

    def _to_depth_meters(self, depth, encoding):
        if encoding == "16UC1":
            depth_m = depth.astype(np.float32) * self.depth_scale
            out_encoding = "16UC1"
        elif encoding == "32FC1":
            depth_m = depth.astype(np.float32)
            out_encoding = "32FC1"
        else:
            raise ValueError("Unsupported depth encoding: {}".format(encoding))
        return depth_m, out_encoding

    def _from_depth_meters(self, depth_m, out_encoding):
        if out_encoding == "16UC1":
            depth_mm = np.clip(depth_m / self.depth_scale, 0.0, np.iinfo(np.uint16).max)
            return depth_mm.astype(np.uint16)
        return depth_m.astype(np.float32)

    def _fill_holes(self, depth_m, valid_mask):
        if self.hole_fill_max_iters <= 0:
            return depth_m

        filled = depth_m.copy()
        holes = ~valid_mask
        if not np.any(holes):
            return filled

        for _ in range(self.hole_fill_max_iters):
            missing = holes & (filled <= 0.0)
            if not np.any(missing):
                break
            dilated = cv2.dilate(filled, self.hole_fill_kernel_mat)
            fillable = missing & (dilated > 0.0)
            if not np.any(fillable):
                break
            filled[fillable] = dilated[fillable]

        return filled

    def on_depth(self, msg):
        try:
            depth = image_msg_to_numpy(msg)
        except ValueError as exc:
            rospy.logerr_throttle(2.0, "Depth refine image decode error: %s", exc)
            return

        if depth.ndim != 2:
            rospy.logerr_throttle(2.0, "Expected single-channel depth image, got shape=%s", depth.shape)
            return

        try:
            depth_m, out_encoding = self._to_depth_meters(depth, msg.encoding)
        except ValueError as exc:
            rospy.logerr_throttle(2.0, "%s", exc)
            return

        valid = np.isfinite(depth_m) & (depth_m >= self.min_depth_m) & (depth_m <= self.max_depth_m)
        proc = np.where(valid, depth_m, 0.0).astype(np.float32)

        # 1) Denoise small speckles while preserving object boundaries.
        if self.enable_median and self.median_ksize >= 3:
            proc = cv2.medianBlur(proc, self.median_ksize)

        # 2) Edge-preserving spatial filter for smoother yet sharp depth transitions.
        if self.enable_bilateral:
            proc = cv2.bilateralFilter(
                proc,
                d=self.bilateral_d,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space,
            )

        # 3) Fill zero/invalid holes using local neighbor propagation.
        if self.enable_hole_filling:
            proc = self._fill_holes(proc, valid)

        # Optional temporal stabilization for frame-to-frame jitter reduction.
        if self.enable_temporal_ema:
            if self.prev_filtered_m is None:
                self.prev_filtered_m = proc.copy()
            else:
                proc = self.temporal_alpha * proc + (1.0 - self.temporal_alpha) * self.prev_filtered_m
                self.prev_filtered_m = proc.copy()

        proc = np.where(np.isfinite(proc), proc, 0.0)
        proc[(proc < self.min_depth_m) | (proc > self.max_depth_m)] = 0.0

        out_depth = self._from_depth_meters(proc, out_encoding)

        try:
            out_msg = numpy_to_image_msg(out_depth, out_encoding, msg.header)
            if not self.preserve_input_header_stamp:
                out_msg.header.stamp = rospy.Time.now()

            if self.latest_camera_info is not None:
                if (
                    self.latest_camera_info.width != out_msg.width
                    or self.latest_camera_info.height != out_msg.height
                ):
                    rospy.logwarn_throttle(
                        5.0,
                        "camera_info resolution (%dx%d) differs from depth image (%dx%d)",
                        self.latest_camera_info.width,
                        self.latest_camera_info.height,
                        out_msg.width,
                        out_msg.height,
                    )

            self.pub.publish(out_msg)
        except Exception as exc:
            rospy.logerr_throttle(2.0, "Depth refine processing error: %s", exc)


def main():
    rospy.init_node("depth_refine_bridge")
    DepthRefineBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
