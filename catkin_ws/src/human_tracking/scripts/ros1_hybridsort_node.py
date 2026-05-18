#!/usr/bin/env python3
# ROS1 Noetic node: subscribe to image topic, run HybridSORT, publish debug image.

import threading
import time
import sys
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import rospy
import torch
import message_filters
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CameraInfo

from trackers.hybrid_sort_tracker.hybrid_sort_reid import Hybrid_Sort_ReID
from trackers.tracking_utils.timer import Timer


def image_msg_to_numpy(msg, desired_encoding=None):
    encoding = msg.encoding
    target = desired_encoding or encoding

    if encoding in ("bgr8", "rgb8"):
        dtype = np.uint8
        channels = 3
    elif encoding in ("mono8", "8UC1"):
        dtype = np.uint8
        channels = 1
    elif encoding == "16UC1":
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
    data = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, row_items))
    image = data[:, : msg.width * channels]
    if channels > 1:
        image = image.reshape((msg.height, msg.width, channels))
    else:
        image = image.reshape((msg.height, msg.width))
    image = np.ascontiguousarray(image)

    if target == "bgr8" and encoding == "rgb8":
        return image[:, :, ::-1].copy()
    if target == "rgb8" and encoding == "bgr8":
        return image[:, :, ::-1].copy()
    if target in (None, "passthrough", encoding, "bgr8") or desired_encoding is None:
        return image
    raise ValueError("Cannot convert image encoding {} to {}".format(encoding, target))


def numpy_to_image_msg(image, encoding, header):
    image = np.ascontiguousarray(image)
    out = Image()
    out.header = header
    out.height = int(image.shape[0])
    out.width = int(image.shape[1])
    out.encoding = encoding
    out.is_bigendian = 0
    out.step = int(image.strides[0])
    out.data = image.tobytes()
    return out


class Predictor:
    def __init__(
        self,
        device,
        test_size=(640, 640),
        detector_weights=None,
        detector_conf=0.1,
        detector_iou=0.7,
        detector_device="cpu",
        detector_classes=None,
        reid_config=None,
        reid_weights=None,
    ):
        self.test_size = test_size
        self.device = device
        self.detector_conf = detector_conf
        self.detector_iou = detector_iou
        self.detector_device = detector_device
        self.detector_weights = detector_weights
        self.detector_classes = HybridSortRosNode._parse_classes_static(detector_classes)
        from ultralytics import YOLO

        self.detector = YOLO(self.detector_weights)

        reid_device = "0" if device.type == "cuda" else "cpu"
        from fast_reid.fast_reid_interfece import FastReIDInterface

        self.encoder = FastReIDInterface(reid_config, reid_weights, reid_device)

    def inference(self, img, timer):
        img_info = {"id": 0, "file_name": None}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        with torch.no_grad():
            timer.tic()
            results = self.detector.predict(
                img,
                conf=self.detector_conf,
                iou=self.detector_iou,
                device=self.detector_device,
                classes=self.detector_classes,
                verbose=False,
            )
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return [None], img_info, np.zeros((0, 1), dtype=np.float32)

            bbox_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            scale = min(self.test_size[0] / float(height), self.test_size[1] / float(width))
            dets = np.concatenate([bbox_xyxy * scale, confs[:, None]], axis=1)
            id_feature = self.encoder.inference(img, bbox_xyxy)
            return [dets], img_info, id_feature


class HybridSortRosNode:
    def __init__(self):
        self.lock = threading.Lock()
        self.busy = False
        self.frame_id = 0
        self.last_debug_image = None
        self.debug_mode = bool(rospy.get_param("~debug_mode", True))
        self.desired_point_topic = rospy.get_param("~desired_point_topic", "/desired_point")
        self.depth_topic = rospy.get_param("~depth_topic", "/d435/depth/image_rect_refined")
        self.depth_info_topic = rospy.get_param("~depth_info_topic", "/d435/depth/camera_info")
        self.target_lost_sec = float(
            rospy.get_param("~target_lost_sec", rospy.get_param("~target_hold_sec", 2.0))
        )
        self.robust_depth_on = bool(rospy.get_param("~robust_depth_on", True))
        self.depth_center_ratio = float(rospy.get_param("~depth_center_ratio", 0.4))
        self.depth_valid_min = float(rospy.get_param("~depth_valid_min", 0.1))
        self.depth_valid_max = float(rospy.get_param("~depth_valid_max", 12.0))
        self.depth_gate_abs_m = float(rospy.get_param("~depth_gate_abs_m", 0.35))
        self.depth_gate_rel = float(rospy.get_param("~depth_gate_rel", 0.25))
        self.depth_reaccept_band_m = float(rospy.get_param("~depth_reaccept_band_m", 0.2))
        self.depth_reaccept_count = int(rospy.get_param("~depth_reaccept_count", 3))
        self.depth_stale_sec = float(rospy.get_param("~depth_stale_sec", 0.5))
        self.depth_kf_on = bool(rospy.get_param("~depth_kf_on", True))
        self.depth_kf_q = float(rospy.get_param("~depth_kf_q", 0.05))
        self.depth_kf_r = float(rospy.get_param("~depth_kf_r", 0.15))
        self.depth_kf_stale_sec = float(rospy.get_param("~depth_kf_stale_sec", 0.5))
        self.current_target_id = None
        self.last_seen_time = None
        self.depth_filter_state = {}
        self.depth_kf_state = {}
        self.fx = rospy.get_param("~fx", 390.2474060058594)
        self.fy = rospy.get_param("~fy", 390.2474060058594)
        self.cx = rospy.get_param("~cx", 316.9652404785156)
        self.cy = rospy.get_param("~cy", 235.58071899414062)

        kiro_root = Path(os.environ.get("KIRO_ROOT", "/opt/kiro"))

        self.image_topic = rospy.get_param("~image_topic", "/d435/color/image_raw")
        self.output_topic = rospy.get_param("~output_topic", "/debug_image")
        self.detector_weights = rospy.get_param(
            "~detector_weights",
            str(kiro_root / "data/models/ultralytics/yolov8n.pt"),
        )
        self.device_name = rospy.get_param("~device", "gpu")
        self.conf = float(rospy.get_param("~conf", 0.1))
        self.nms = float(rospy.get_param("~nms", 0.7))
        self.detector_classes = rospy.get_param("~detector_classes", [0])
        self.detector_classes = self._parse_classes_static(self.detector_classes)
        self.tsize = int(rospy.get_param("~tsize", 640))
        self.test_size = (self.tsize, self.tsize)

        self.track_thresh = float(rospy.get_param("~track_thresh", 0.5))
        self.iou_thresh = float(rospy.get_param("~iou_thresh", 0.3))
        self.deltat = int(rospy.get_param("~deltat", 3))
        self.inertia = float(rospy.get_param("~inertia", 0.05))
        self.asso = rospy.get_param("~asso", "Height_Modulated_IoU")
        self.asso = self._normalize_asso_name(self.asso)
        self.use_byte = bool(rospy.get_param("~use_byte", True))
        self.min_box_area = float(rospy.get_param("~min_box_area", 10))
        self.TCM_first_step = bool(rospy.get_param("~TCM_first_step", True))
        self.TCM_byte_step = bool(rospy.get_param("~TCM_byte_step", True))
        self.TCM_first_step_weight = float(rospy.get_param("~TCM_first_step_weight", 1.0))
        self.TCM_byte_step_weight = float(rospy.get_param("~TCM_byte_step_weight", 1.0))

        self.fast_reid_config = rospy.get_param(
            "~fast_reid_config",
            str(kiro_root / "HybridSORT/fast_reid/configs/MSMT17/sbs_R50-ibn.yml"),
        )
        self.fast_reid_weights = rospy.get_param(
            "~fast_reid_weights",
            str(kiro_root / "HybridSORT/fast_reid/logs/msmt17/sbs_R50-ibn.pth"),
        )
        self.EG_weight_high_score = float(rospy.get_param("~EG_weight_high_score", 4.6))
        self.EG_weight_low_score = float(rospy.get_param("~EG_weight_low_score", 1.3))
        self.low_thresh = float(rospy.get_param("~low_thresh", 0.1))
        self.high_score_matching_thresh = float(rospy.get_param("~high_score_matching_thresh", 0.7))
        self.low_score_matching_thresh = float(rospy.get_param("~low_score_matching_thresh", 0.5))
        self.alpha = float(rospy.get_param("~alpha", 0.9))
        self.with_longterm_reid = bool(rospy.get_param("~with_longterm_reid", True))
        self.longterm_reid_weight = float(rospy.get_param("~longterm_reid_weight", 0.0))
        self.with_longterm_reid_correction = bool(rospy.get_param("~with_longterm_reid_correction", True))
        self.longterm_reid_correction_thresh = float(
            rospy.get_param("~longterm_reid_correction_thresh", 0.4)
        )
        self.longterm_reid_correction_thresh_low = float(
            rospy.get_param("~longterm_reid_correction_thresh_low", 0.4)
        )
        self.longterm_bank_length = int(rospy.get_param("~longterm_bank_length", 30))
        self.confidence_weighted_longterm_reid = bool(
            rospy.get_param("~confidence_weighted_longterm_reid", False)
        )
        self.longterm_reid_conf_gamma = float(rospy.get_param("~longterm_reid_conf_gamma", 1.0))
        self.adapfs = bool(rospy.get_param("~adapfs", False))
        self.ECC = bool(rospy.get_param("~ECC", True))
        self.dataset = rospy.get_param("~dataset", "")

        self.repo_root = Path(__file__).resolve().parent
        self.workspace_root = self.repo_root.parent
        self.detector_weights = self._resolve_path(self.detector_weights)
        self.fast_reid_config = self._resolve_path(self.fast_reid_config)
        self.fast_reid_weights = self._resolve_path(self.fast_reid_weights)

        self.device = self._resolve_device(self.device_name)
        rospy.loginfo("Detector: ultralytics (%s)", self.detector_weights)

        self.predictor = Predictor(
            self.device,
            test_size=self.test_size,
            detector_weights=str(self.detector_weights),
            detector_conf=self.conf,
            detector_iou=self.nms,
            detector_device="0" if self.device.type == "cuda" else "cpu",
            detector_classes=self.detector_classes,
            reid_config=str(self.fast_reid_config),
            reid_weights=str(self.fast_reid_weights),
        )

        self.tracker_args = SimpleNamespace(
            track_thresh=self.track_thresh,
            use_byte=self.use_byte,
            TCM_first_step=self.TCM_first_step,
            TCM_byte_step=self.TCM_byte_step,
            TCM_first_step_weight=self.TCM_first_step_weight,
            TCM_byte_step_weight=self.TCM_byte_step_weight,
            EG_weight_high_score=self.EG_weight_high_score,
            EG_weight_low_score=self.EG_weight_low_score,
            low_thresh=self.low_thresh,
            high_score_matching_thresh=self.high_score_matching_thresh,
            low_score_matching_thresh=self.low_score_matching_thresh,
            alpha=self.alpha,
            with_longterm_reid=self.with_longterm_reid,
            longterm_reid_weight=self.longterm_reid_weight,
            with_longterm_reid_correction=self.with_longterm_reid_correction,
            longterm_reid_correction_thresh=self.longterm_reid_correction_thresh,
            longterm_reid_correction_thresh_low=self.longterm_reid_correction_thresh_low,
            longterm_bank_length=self.longterm_bank_length,
            confidence_weighted_longterm_reid=self.confidence_weighted_longterm_reid,
            longterm_reid_conf_gamma=self.longterm_reid_conf_gamma,
            adapfs=self.adapfs,
            ECC=self.ECC,
            dataset=self.dataset,
        )

        self.tracker = Hybrid_Sort_ReID(
            self.tracker_args,
            det_thresh=self.track_thresh,
            iou_threshold=self.iou_thresh,
            asso_func=self.asso,
            delta_t=self.deltat,
            inertia=self.inertia,
        )

        self.timer = Timer()
        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=1) if self.debug_mode else None
        self.point_pub = rospy.Publisher(self.desired_point_topic, PointStamped, queue_size=1)

        self.depth_info_sub = rospy.Subscriber(self.depth_info_topic, CameraInfo, self.on_depth_info, queue_size=1)
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub],
            queue_size=5,
            slop=0.05,
        )
        self.sync.registerCallback(self.on_image_depth)
        rospy.loginfo("HybridSORT ROS node ready. Sub: %s -> Pub: %s", self.image_topic, self.output_topic)

    def _resolve_path(self, value):
        if not value:
            return value
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.repo_root / path).resolve()

    @staticmethod
    def _parse_classes_static(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            if stripped.lower() in ("none", "all"):
                return None
            cleaned = stripped.strip("[]")
            parts = [p for p in cleaned.replace(",", " ").split() if p]
            return [int(p) for p in parts]
        return None

    @staticmethod
    def _normalize_asso_name(value):
        if value is None:
            return "iou"
        asso = str(value).strip()
        lower = asso.lower()
        if lower == "hmiou":
            return "Height_Modulated_IoU"
        if lower in ("height_modulated_iou", "height-modulated-iou"):
            return "Height_Modulated_IoU"
        if lower in ("iou", "giou", "ciou", "diou", "ct_dist"):
            return lower
        return asso

    def _resolve_device(self, value):
        if value == "gpu":
            if torch.cuda.is_available():
                return torch.device("cuda")
            rospy.logwarn("CUDA not available. Falling back to CPU.")
        return torch.device("cpu")

    def on_depth_info(self, msg):
        if not msg.K or len(msg.K) < 9:
            return
        self.fx = float(msg.K[0])
        self.fy = float(msg.K[4])
        self.cx = float(msg.K[2])
        self.cy = float(msg.K[5])

    def _depth_to_meters(self, depth_val, encoding):
        if encoding == "16UC1":
            return float(depth_val) / 1000.0
        if encoding in ("32FC1", "32FC"):
            return float(depth_val)
        return None

    def _get_depth_at(self, depth_img, u, v, encoding):
        h, w = depth_img.shape[:2]
        if u < 0 or v < 0 or u >= w or v >= h:
            return None
        depth_val = depth_img[v, u]
        z = self._depth_to_meters(depth_val, encoding)
        if z and z > 0:
            return z
        u0, u1 = max(0, u - 1), min(w - 1, u + 1)
        v0, v1 = max(0, v - 1), min(h - 1, v + 1)
        window = depth_img[v0 : v1 + 1, u0 : u1 + 1].reshape(-1)
        vals = []
        for dv in window:
            zc = self._depth_to_meters(dv, encoding)
            if zc and zc > 0:
                vals.append(zc)
        if not vals:
            return None
        return float(np.median(vals))

    def _roi_median_depth(self, depth_img, x1, y1, x2, y2, encoding):
        h, w = depth_img.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))
        if x2 < x1 or y2 < y1:
            return None

        roi = depth_img[y1 : y2 + 1, x1 : x2 + 1]
        if roi.size == 0:
            return None

        if encoding == "16UC1":
            vals = roi.astype(np.float32).reshape(-1) / 1000.0
        elif encoding in ("32FC1", "32FC"):
            vals = roi.astype(np.float32).reshape(-1)
        else:
            return None

        valid = vals[np.isfinite(vals)]
        valid = valid[(valid > self.depth_valid_min) & (valid < self.depth_valid_max)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _stabilize_depth(self, target_id, raw_z, stamp):
        state = self.depth_filter_state.get(target_id)
        now_sec = stamp.to_sec()

        if raw_z is None:
            if state is None:
                return None
            if now_sec - state["accepted_time"] <= self.depth_stale_sec:
                return state["accepted"]
            return None

        if state is None:
            self.depth_filter_state[target_id] = {
                "accepted": raw_z,
                "accepted_time": now_sec,
                "candidate": None,
                "candidate_count": 0,
            }
            return raw_z

        accepted = state["accepted"]
        gate = max(self.depth_gate_abs_m, abs(accepted) * self.depth_gate_rel)
        if abs(raw_z - accepted) <= gate:
            state["accepted"] = raw_z
            state["accepted_time"] = now_sec
            state["candidate"] = None
            state["candidate_count"] = 0
            return raw_z

        cand = state["candidate"]
        if cand is not None and abs(raw_z - cand) <= self.depth_reaccept_band_m:
            state["candidate_count"] += 1
            state["candidate"] = (cand * 0.5) + (raw_z * 0.5)
        else:
            state["candidate"] = raw_z
            state["candidate_count"] = 1

        if state["candidate_count"] >= self.depth_reaccept_count:
            state["accepted"] = state["candidate"]
            state["accepted_time"] = now_sec
            state["candidate"] = None
            state["candidate_count"] = 0
            return state["accepted"]

        if now_sec - state["accepted_time"] <= self.depth_stale_sec:
            return accepted
        return None

    def _apply_depth_kalman(self, target_id, stable_z, stamp):
        if not self.depth_kf_on:
            return stable_z

        now_sec = stamp.to_sec()
        state = self.depth_kf_state.get(target_id)

        if stable_z is None:
            if state is None:
                return None
            if now_sec - state["time"] <= self.depth_kf_stale_sec:
                return float(state["x"][0, 0])
            return None

        if state is None:
            x = np.array([[float(stable_z)], [0.0]], dtype=np.float32)
            P = np.eye(2, dtype=np.float32)
            self.depth_kf_state[target_id] = {
                "x": x,
                "P": P,
                "time": now_sec,
            }
            return float(stable_z)

        dt = max(1e-3, min(1.0, now_sec - state["time"]))
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
        H = np.array([[1.0, 0.0]], dtype=np.float32)
        q = max(1e-6, self.depth_kf_q)
        r = max(1e-6, self.depth_kf_r)
        Q = q * np.array(
            [
                [0.25 * dt * dt * dt * dt, 0.5 * dt * dt * dt],
                [0.5 * dt * dt * dt, dt * dt],
            ],
            dtype=np.float32,
        )
        R = np.array([[r]], dtype=np.float32)

        x_pred = F @ state["x"]
        P_pred = F @ state["P"] @ F.T + Q

        z = np.array([[float(stable_z)]], dtype=np.float32)
        y = z - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_new = x_pred + K @ y
        P_new = (np.eye(2, dtype=np.float32) - K @ H) @ P_pred

        state["x"] = x_new
        state["P"] = P_new
        state["time"] = now_sec
        return float(x_new[0, 0])

    def _get_tlwh_depth(self, tlwh, img_shape, depth_img, encoding):
        x, y, w, h = tlwh
        u_img = int(round(x + w / 2.0))
        v_img = int(round(y + h / 2.0))
        img_h, img_w = img_shape[:2]
        depth_h, depth_w = depth_img.shape[:2]
        if img_w > 0 and img_h > 0:
            scale_x = depth_w / float(img_w)
            scale_y = depth_h / float(img_h)
            x1 = int(round(x * scale_x))
            y1 = int(round(y * scale_y))
            x2 = int(round((x + w) * scale_x))
            y2 = int(round((y + h) * scale_y))
            u = int(round(u_img * scale_x))
            v = int(round(v_img * scale_y))
        else:
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + w))
            y2 = int(round(y + h))
            u, v = u_img, v_img

        if self.robust_depth_on:
            roi_w = max(1, x2 - x1 + 1)
            roi_h = max(1, y2 - y1 + 1)
            c_ratio = max(0.05, min(1.0, self.depth_center_ratio))
            cx1 = x1 + int((1.0 - c_ratio) * roi_w * 0.5)
            cx2 = x2 - int((1.0 - c_ratio) * roi_w * 0.5)
            cy1 = y1 + int((1.0 - c_ratio) * roi_h * 0.5)
            cy2 = y2 - int((1.0 - c_ratio) * roi_h * 0.5)

            z = self._roi_median_depth(depth_img, cx1, cy1, cx2, cy2, encoding)
            if z is None:
                z = self._get_depth_at(depth_img, u, v, encoding)
        else:
            z = self._get_depth_at(depth_img, u, v, encoding)
        return z, u, v

    def _select_closest_target(self, ids, tlwhs, img_shape, depth_img, encoding):
        best_tid = None
        best_depth = None
        for tid in ids:
            tlwh = tlwhs.get(tid)
            if tlwh is None:
                continue
            z, _, _ = self._get_tlwh_depth(tlwh, img_shape, depth_img, encoding)
            if z is None or z <= 0:
                continue
            if best_depth is None or z < best_depth:
                best_depth = z
                best_tid = tid
        if best_tid is not None:
            return best_tid
        return min(ids) if ids else None

    def _set_current_target(self, target_id, stamp):
        if self.current_target_id != target_id:
            self.current_target_id = target_id
        if target_id is not None:
            self.last_seen_time = stamp

    def on_image_depth(self, img_msg, depth_msg):
        if not self.lock.acquire(False):
            return
        if self.busy:
            self.lock.release()
            return
        self.busy = True
        self.lock.release()

        try:
            frame_start = time.time()
            img = image_msg_to_numpy(img_msg, desired_encoding="bgr8")
            depth_img = image_msg_to_numpy(depth_msg, desired_encoding="passthrough")

            outputs, img_info, id_feature = self.predictor.inference(img, self.timer)

            target_tlwh = None
            target_id = None
            all_ids = []
            all_tlwhs = {}
            if outputs and outputs[0] is not None:
                online_targets = self.tracker.update(
                    outputs[0],
                    [img_info["height"], img_info["width"]],
                    self.test_size,
                    id_feature=id_feature,
                )

                ids = []
                tlwhs = {}
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    if tlwh[2] * tlwh[3] > self.min_box_area:
                        tid = int(t[4])
                        ids.append(tid)
                        tlwhs[tid] = tlwh
                all_ids = ids
                all_tlwhs = tlwhs

                now = rospy.Time.now()
                if ids:
                    if self.current_target_id is None:
                        new_target_id = self._select_closest_target(
                            ids, tlwhs, img.shape, depth_img, depth_msg.encoding
                        )
                        self._set_current_target(new_target_id, now)
                    elif self.current_target_id in ids:
                        self.last_seen_time = now
                    else:
                        if self.last_seen_time is None:
                            lost_elapsed = float("inf")
                        else:
                            lost_elapsed = (now - self.last_seen_time).to_sec()
                        if lost_elapsed >= self.target_lost_sec:
                            new_target_id = self._select_closest_target(
                                ids, tlwhs, img.shape, depth_img, depth_msg.encoding
                            )
                            self._set_current_target(new_target_id, now)

                if self.current_target_id is not None and self.current_target_id in tlwhs:
                    target_id = self.current_target_id
                    target_tlwh = tlwhs[target_id]

                if self.current_target_id is not None and self.last_seen_time is not None:
                    if (now - self.last_seen_time).to_sec() < self.target_lost_sec and (
                        self.current_target_id not in ids
                    ):
                        target_id = None
                        target_tlwh = None

            if target_tlwh is not None and self.fx and self.fy:
                z_raw, u, v = self._get_tlwh_depth(target_tlwh, img.shape, depth_img, depth_msg.encoding)
                if self.robust_depth_on:
                    z_stable = self._stabilize_depth(target_id, z_raw, depth_msg.header.stamp)
                    z = self._apply_depth_kalman(target_id, z_stable, depth_msg.header.stamp)
                else:
                    z = z_raw
                #rospy.loginfo("Target ID: %s at (u=%d, v=%d) with depth z=%.3f m", target_id, u, v, z if z else -1.0)
                if z is not None and z > 0:
                    X = (u - self.cx) * z / self.fx
                    Y = (v - self.cy) * z / self.fy
                    pt_msg = PointStamped()
                    pt_msg.header = depth_msg.header
                    pt_msg.point.x = float(X)
                    pt_msg.point.y = float(Y)
                    pt_msg.point.z = float(z)
                    #rospy.loginfo("HERE!!!")
                    self.point_pub.publish(pt_msg)

            if self.debug_mode:
                online_im = img_info["raw_img"].copy()
                fps = 1.0 / max(1e-5, time.time() - frame_start)
                cv2.putText(
                    online_im,
                    f"frame: {self.frame_id} fps: {fps:.2f} num: {len(all_ids)}",
                    (0, 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 0, 255),
                    2,
                )
                for tid in all_ids:
                    tlwh = all_tlwhs.get(tid)
                    if not tlwh:
                        continue
                    x, y, w, h = tlwh
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    is_target = tid == target_id
                    color = (0, 0, 255) if is_target else (255, 255, 255)
                    thickness = 3 if is_target else 1
                    cv2.rectangle(online_im, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(
                        online_im,
                        str(tid),
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        color,
                        2,
                    )
                out_msg = numpy_to_image_msg(online_im, "bgr8", img_msg.header)
                self.pub.publish(out_msg)
                self.last_debug_image = online_im

            self.frame_id += 1
        except Exception as exc:
            rospy.logerr("HybridSORT processing failed: %s", exc)
            rospy.logerr("%s", traceback.format_exc())
        finally:
            self.lock.acquire(False)
            self.busy = False
            self.lock.release()


def main():
    rospy.init_node("hybridsort_tracker")
    HybridSortRosNode()
    rospy.spin()


if __name__ == "__main__":
    main()
