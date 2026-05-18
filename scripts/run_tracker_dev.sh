#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${IMAGE:-kiro-visiontracker:v2}"
USE_GPU="${USE_GPU:-auto}"

GPU_ARGS=()
if [[ "${USE_GPU}" == "1" || "${USE_GPU}" == "true" || "${USE_GPU}" == "yes" ]]; then
  GPU_ARGS=(--gpus all)
elif [[ "${USE_GPU}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi 'nvidia'; then
    GPU_ARGS=(--gpus all)
  fi
fi

docker run --rm -it \
  "${GPU_ARGS[@]}" \
  --net host \
  --ipc host \
  -e ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}" \
  -e ROS_IP="${ROS_IP:-127.0.0.1}" \
  -v "${DEPLOY_DIR}:/opt/kiro" \
  "${IMAGE}" \
  bash -lc 'cd /opt/kiro/catkin_ws && source /opt/ros/noetic/setup.bash && catkin_make >/tmp/kiro_catkin_make.log && source devel/setup.bash && roslaunch human_tracking ros1_hybridsort_deploy.launch "$@"' _ "$@"
