#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
AI_MODULE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DST="$SCRIPT_DIR/models"
MODELS_SRC="${GROUNDINGDINO_MODELS:-$HOME/groundingdino_models}"
IMAGE_TAG="${IMAGE_TAG:-iros2026/ai_module:latest}"
TAR_OUT="${TAR_OUT:-/var/tmp/iros2026_ai_module.tar}"

mkdir -p "$MODELS_DST"

if [ -f "$MODELS_SRC/groundingdino_swint_ogc.pth" ]; then
  echo "Staging GroundingDINO weights from $MODELS_SRC"
  cp -f "$MODELS_SRC/groundingdino_swint_ogc.pth" "$MODELS_DST/groundingdino_swint_ogc.pth"
else
  echo "No local weights at $MODELS_SRC/groundingdino_swint_ogc.pth — Dockerfile will wget them."
fi

cd "$AI_MODULE_ROOT"
echo "Building $IMAGE_TAG → $TAR_OUT (single tag; docker-container --load cannot alias two names)"
rm -f "$TAR_OUT"
docker build \
  -t "$IMAGE_TAG" \
  -o "type=docker,dest=$TAR_OUT" \
  -f docker/Dockerfile .
echo "Loading $TAR_OUT into the local docker engine..."
docker load -i "$TAR_OUT"
docker tag "$IMAGE_TAG" docker-ai_module:latest
rm -f "$TAR_OUT"
echo "Built $IMAGE_TAG and docker-ai_module:latest"
