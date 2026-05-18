# KIRO VisionTracker v2

Minimal deploy bundle for running `ros1_hybridsort_node.py` through ROS Noetic.

This `KIRO_VisionTracker_v2` folder intentionally excludes rosbag files. The launch file expects live or externally replayed ROS topics:

- RGB image: `/d435/color/image_raw`
- Raw depth: `/d435/depth/image_rect_raw`
- Depth camera info: `/d435/depth/camera_info`
- Refined depth output: `/d435/depth/image_rect_refined`
- Tracker debug image: `/debug_image`
- Target 3D point: `/desired_point`

## Build

```bash
cd KIRO_VisionTracker_v2
./scripts/build_image.sh
```

The image builds with CPU PyTorch by default. To build a CUDA 11.8 PyTorch image for an NVIDIA Docker host:

```bash
TORCH_FLAVOR=cu118 ./scripts/build_image.sh
```

## Required Model Files

Model weights are intentionally not tracked in git. Place them before building:

```bash
data/models/ultralytics/yolov8n.pt
HybridSORT/fast_reid/logs/msmt17/sbs_R50-ibn.pth
```

The `.dockerignore` keeps these local files available to Docker builds, while `.gitignore` keeps them out of GitHub.

If you only want to build the Docker environment before copying model files, skip model loading validation:

```bash
SKIP_MODEL_VALIDATE=1 ./scripts/build_image.sh
```

That image is useful for checking dependencies, but `run_tracker.sh` still needs the model files baked into the image or provided through the dev-mounted folder.

## Run

```bash
./scripts/run_tracker.sh
```

`run_tracker.sh` uses the GPU only when Docker/NVIDIA support is detected. Force CPU mode with:

```bash
USE_GPU=0 ./scripts/run_tracker.sh device:=cpu
```

Force GPU mode with:

```bash
USE_GPU=1 ./scripts/run_tracker.sh device:=gpu
```

Override topics or parameters by passing normal roslaunch args:

```bash
./scripts/run_tracker.sh image_topic:=/camera/color/image_raw depth_raw_topic:=/camera/depth/image_rect_raw
```

## Rebuilds

You do not need to rebuild the image for every run. Rebuild after changing the Dockerfile, installed dependencies, bundled models, or files that must be baked into the portable image.

To work freely inside the Docker environment with this folder mounted:

```bash
./scripts/shell.sh
```

The shell script rebuilds only the mounted catkin workspace, sources ROS, and drops you into `/opt/kiro`. You can also run one command inside the same environment:

```bash
./scripts/shell.sh rostopic list
```

Use the baked image without mounting local files with:

```bash
MOUNT_DEPLOY=0 ./scripts/shell.sh
```

For small code or launch edits during development, use the dev runner. It mounts this folder into the container and rebuilds only the catkin workspace:

```bash
./scripts/run_tracker_dev.sh device:=cpu
```

After the change works, run `./scripts/build_image.sh` once to bake the latest folder into the portable image.

## Launch

The deploy entrypoint runs:

```bash
roslaunch human_tracking ros1_hybridsort_deploy.launch
```

`ros1_hybridsort_deploy.launch` starts `depth_refine_bridge` and `hybridsort_tracker`.
It does not start `rosbag play`.

## Optional Rosbag Playback

Rosbags are not bundled in this folder. To replay an external bag in a separate terminal:

```bash
./scripts/play_bag.sh /path/to/file.bag --clock -r 0.5
```

The script mounts the bag directory read-only into the same Docker image and runs `rosbag play`.

## Debug Image Snapshot

The deploy image does not bundle a GUI viewer. Save one `/debug_image` frame from a separate terminal:

```bash
./scripts/view_debug_image.sh
```

Equivalent manual command:

```bash
./scripts/shell.sh python3 /opt/kiro/scripts/debug_image_snapshot.py /debug_image /opt/kiro/debug_image.jpg
```

If the laptop is too slow for realtime bag playback, use a lower rosbag rate:

```bash
./scripts/play_bag.sh /path/to/file.bag -r 0.5
```

## ReID Model

The deploy launch is trimmed for FastReID only:

```bash
fast_reid_config:=/opt/kiro/HybridSORT/fast_reid/configs/MSMT17/sbs_R50-ibn.yml
fast_reid_weights:=/opt/kiro/HybridSORT/fast_reid/logs/msmt17/sbs_R50-ibn.pth
```
