#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCH_FLAVOR="${TORCH_FLAVOR:-cpu}"
NO_CACHE="${NO_CACHE:-0}"
SKIP_MODEL_VALIDATE="${SKIP_MODEL_VALIDATE:-0}"

BUILD_ARGS=()
if [[ "${NO_CACHE}" == "1" || "${NO_CACHE}" == "true" || "${NO_CACHE}" == "yes" ]]; then
  BUILD_ARGS+=(--no-cache)
fi

docker build \
  "${BUILD_ARGS[@]}" \
  --build-arg "TORCH_FLAVOR=${TORCH_FLAVOR}" \
  --build-arg "SKIP_MODEL_VALIDATE=${SKIP_MODEL_VALIDATE}" \
  -t kiro-visiontracker:v2 \
  "${DEPLOY_DIR}"
