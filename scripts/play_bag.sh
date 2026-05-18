#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/file.bag [rosbag play args...]" >&2
  echo "Example: $0 /data/test.bag --clock -r 0.5" >&2
  exit 2
fi

IMAGE="${IMAGE:-kiro-visiontracker:v2}"
BAG_PATH="$(realpath "$1")"
shift

if [[ ! -f "${BAG_PATH}" ]]; then
  echo "Bag file not found: ${BAG_PATH}" >&2
  exit 1
fi

BAG_DIR="$(dirname "${BAG_PATH}")"
BAG_FILE="$(basename "${BAG_PATH}")"

docker run --rm -it \
  --net host \
  --ipc host \
  -e ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}" \
  -e ROS_IP="${ROS_IP:-127.0.0.1}" \
  -v "${BAG_DIR}:/bags:ro" \
  "${IMAGE}" \
  rosbag play "$@" "/bags/${BAG_FILE}"
