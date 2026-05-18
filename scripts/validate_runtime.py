#!/usr/bin/env python3

import sys
import torch
import torchvision

sys.path.insert(0, "/opt/kiro/HybridSORT")

from fast_reid.fast_reid_interfece import FastReIDInterface
from trackers.hybrid_sort_tracker.hybrid_sort_reid import Hybrid_Sort_ReID  # noqa: F401
from trackers.tracking_utils.timer import Timer  # noqa: F401
from ultralytics import YOLO


if torch.__version__.split("+", 1)[0] != "2.1.2":
    raise RuntimeError("Unexpected torch version: {}".format(torch.__version__))
if torchvision.__version__.split("+", 1)[0] != "0.16.2":
    raise RuntimeError("Unexpected torchvision version: {}".format(torchvision.__version__))


def main():
    YOLO("/opt/kiro/data/models/ultralytics/yolov8n.pt")
    FastReIDInterface(
        "/opt/kiro/HybridSORT/fast_reid/configs/MSMT17/sbs_R50-ibn.yml",
        "/opt/kiro/HybridSORT/fast_reid/logs/msmt17/sbs_R50-ibn.pth",
        "cpu",
    )
    print("Deploy Python import/model validation OK")


if __name__ == "__main__":
    main()
