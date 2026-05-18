#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${IMAGE:-kiro-visiontracker:v2}"
USE_GPU="${USE_GPU:-auto}"
MOUNT_DEPLOY="${MOUNT_DEPLOY:-1}"
ENABLE_GUI="${ENABLE_GUI:-auto}"

GPU_ARGS=()
if [[ "${USE_GPU}" == "1" || "${USE_GPU}" == "true" || "${USE_GPU}" == "yes" ]]; then
  GPU_ARGS=(--gpus all)
elif [[ "${USE_GPU}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi 'nvidia'; then
    GPU_ARGS=(--gpus all)
  fi
fi

VOLUME_ARGS=()
TTY_ARGS=(-i)
if [[ -t 0 ]]; then
  TTY_ARGS=(-it)
fi

GUI_ARGS=()
if [[ "${ENABLE_GUI}" == "1" || "${ENABLE_GUI}" == "true" || "${ENABLE_GUI}" == "yes" ]]; then
  ENABLE_GUI="enabled"
elif [[ "${ENABLE_GUI}" == "auto" && -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
  ENABLE_GUI="enabled"
fi

if [[ "${ENABLE_GUI}" == "enabled" ]]; then
  GUI_ARGS=(
    -e "DISPLAY=${DISPLAY}"
    -e QT_X11_NO_MITSHM=1
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  )
  if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    GUI_ARGS+=(-e "XAUTHORITY=/tmp/.docker.xauth" -v "${XAUTHORITY}:/tmp/.docker.xauth:ro")
  fi
fi

SETUP_CMD='source /opt/ros/noetic/setup.bash'
if [[ "${MOUNT_DEPLOY}" == "1" || "${MOUNT_DEPLOY}" == "true" || "${MOUNT_DEPLOY}" == "yes" ]]; then
  VOLUME_ARGS=(-v "${DEPLOY_DIR}:/opt/kiro")
  SETUP_CMD="${SETUP_CMD}"'
cd /opt/kiro/catkin_ws
mkdir -p /tmp/kiro_catkin_lock
flock /tmp/kiro_catkin_lock/lock catkin_make >/tmp/kiro_catkin_make.log
source /opt/kiro/catkin_ws/devel/setup.bash'
else
  SETUP_CMD="${SETUP_CMD}"'
source /opt/kiro/catkin_ws/devel/setup.bash'
fi

if [[ "$#" -gt 0 ]]; then
  USER_CMD='exec "$@"'
else
  USER_CMD='exec bash -l'
fi

docker run --rm "${TTY_ARGS[@]}" \
  "${GPU_ARGS[@]}" \
  --net host \
  --ipc host \
  -e ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}" \
  -e ROS_IP="${ROS_IP:-127.0.0.1}" \
  "${GUI_ARGS[@]}" \
  "${VOLUME_ARGS[@]}" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc 'set -e
'"${SETUP_CMD}"'
export KIRO_ROOT="${KIRO_ROOT:-/opt/kiro}"
export PYTHONPATH="/opt/kiro/HybridSORT:${PYTHONPATH:-}"
cd /opt/kiro
'"${USER_CMD}" _ "$@"
