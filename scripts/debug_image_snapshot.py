#!/usr/bin/env python3

import sys

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image


def image_msg_to_numpy(msg):
    if msg.encoding in ("bgr8", "rgb8"):
        dtype = np.uint8
        channels = 3
    elif msg.encoding in ("mono8", "8UC1"):
        dtype = np.uint8
        channels = 1
    else:
        raise ValueError("Unsupported debug image encoding: {}".format(msg.encoding))

    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    row_items = msg.step // dtype.itemsize
    data = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, row_items))
    image = data[:, : msg.width * channels]
    if channels > 1:
        image = image.reshape((msg.height, msg.width, channels))
    else:
        image = image.reshape((msg.height, msg.width))
    image = np.ascontiguousarray(image)
    if msg.encoding == "rgb8":
        image = image[:, :, ::-1].copy()
    return image


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "/debug_image"
    output = sys.argv[2] if len(sys.argv) > 2 else "/opt/kiro/debug_image.jpg"
    rospy.init_node("debug_image_snapshot", anonymous=True)
    msg = rospy.wait_for_message(topic, Image, timeout=10.0)
    image = image_msg_to_numpy(msg)
    if not cv2.imwrite(output, image):
        raise RuntimeError("Failed to write {}".format(output))
    print(output)


if __name__ == "__main__":
    main()
