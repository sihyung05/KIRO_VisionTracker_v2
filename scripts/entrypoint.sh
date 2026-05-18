#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash
source /opt/kiro/catkin_ws/devel/setup.bash

export KIRO_ROOT="${KIRO_ROOT:-/opt/kiro}"
export PYTHONPATH="/opt/kiro/HybridSORT:${PYTHONPATH}"

exec "$@"
