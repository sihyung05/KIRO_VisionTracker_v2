#!/usr/bin/env python3

import os
from pathlib import Path
import sys
import torch
import torchvision

sys.path.insert(0, "/opt/kiro/HybridSORT")

from fast_reid.fast_reid_interfece import FastReIDInterface
from trackers.hybrid_sort_tracker.hybrid_sort_reid import Hybrid_Sort_ReID  # noqa: F401
from trackers.tracking_utils.timer import Timer  # noqa: F401
from ultralytics import YOLO

YOLO_WEIGHTS = Path("/opt/kiro/data/models/ultralytics/yolov8n.pt")
REID_CONFIG = Path("/opt/kiro/HybridSORT/fast_reid/configs/MSMT17/sbs_R50-ibn.yml")
REID_WEIGHTS = Path("/opt/kiro/HybridSORT/fast_reid/logs/msmt17/sbs_R50-ibn.pth")


if torch.__version__.split("+", 1)[0] != "2.1.2":
    raise RuntimeError("Unexpected torch version: {}".format(torch.__version__))
if torchvision.__version__.split("+", 1)[0] != "0.16.2":
    raise RuntimeError("Unexpected torchvision version: {}".format(torchvision.__version__))


def main():
    missing = [path for path in (YOLO_WEIGHTS, REID_CONFIG, REID_WEIGHTS) if not path.exists()]
    skip_model_validate = os.environ.get("SKIP_MODEL_VALIDATE", "0").lower() in ("1", "true", "yes")
    if missing:
        message = (
            "Missing required model/config files:\n"
            + "\n".join("  - {}".format(path) for path in missing)
            + "\n\nPlace the model files in the repo before building, or run:\n"
            + "  SKIP_MODEL_VALIDATE=1 ./scripts/build_image.sh\n"
            + "to build the environment image without baked model validation."
        )
        if skip_model_validate:
            print(message)
            print("Skipping model load validation because SKIP_MODEL_VALIDATE=1")
            return
        raise FileNotFoundError(message)

    YOLO(str(YOLO_WEIGHTS))
    FastReIDInterface(
        str(REID_CONFIG),
        str(REID_WEIGHTS),
        "cpu",
    )
    print("Deploy Python import/model validation OK")


if __name__ == "__main__":
    main()
