#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPIC="${1:-/debug_image}"
OUTPUT="${2:-/opt/kiro/debug_image.jpg}"

exec "${SCRIPT_DIR}/shell.sh" python3 /opt/kiro/scripts/debug_image_snapshot.py "${TOPIC}" "${OUTPUT}"
