#!/usr/bin/env bash
# A/B GroundingDINO vs OWL-ViT v2 on the same Unity scene.
#
# Inside iros2026_ai_module, with sim + autonomy already up:
#   bash /home/docker/ai_module/src/vlm_pipeline_live/scripts/compare_dino_owlvit.sh
#
# Run each launch to completion (explorer or manual). Then this script
# scores both latest graphs against Unity GT.

set -euo pipefail

ROOT="${AI_MODULE_ROOT:-/home/docker/ai_module}"
SCENE="${SCENE_NAME:-arabic_room}"
GT="${GT_SCENE_GRAPH:-/home/docker/vla3d_data/Unity/${SCENE}/${SCENE}_scene_graph.json}"
CAPTURE_ROOT="${CAPTURE_ROOT:-/tmp/vlm_live_captures}"
OUT="${COMPARE_OUT:-/tmp/${SCENE}_dino_vs_owlvit.json}"

source /opt/ros/jazzy/setup.bash
if [[ -f "${ROOT}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/install/setup.bash"
fi

cat <<EOF
============================================================
 GroundingDINO vs OWL-ViT v2  (scene=${SCENE})
 GT: ${GT}
============================================================

# 1) GroundingDINO (explorer)
ros2 launch vlm_pipeline_live pipeline_b.launch.py \\
  scene_name:=${SCENE}_dino scene_type:=${SCENE} \\
  detector_backend:=grounding_dino box_threshold:=0.35

# 2) OWL-ViT v2 (same scene / similar tour)
# First run downloads google/owlv2-base-patch16 from Hugging Face.
ros2 launch vlm_pipeline_live pipeline_b.launch.py \\
  scene_name:=${SCENE}_owlvit scene_type:=${SCENE} \\
  detector_backend:=owlvit box_threshold:=0.2

After both graphs exist, score them:

ros2 run vlm_pipeline_live compare_backend_runs -- \\
  --gt ${GT} \\
  --run grounding_dino:${CAPTURE_ROOT}/${SCENE}_dino/latest_scene_graph.json \\
  --run owlvit:${CAPTURE_ROOT}/${SCENE}_owlvit/latest_scene_graph.json \\
  --out ${OUT}

EOF

DINO="${CAPTURE_ROOT}/${SCENE}_dino/latest_scene_graph.json"
OWL="${CAPTURE_ROOT}/${SCENE}_owlvit/latest_scene_graph.json"
if [[ -f "${DINO}" && -f "${OWL}" && -f "${GT}" ]]; then
  echo "Both captures found — running compare now."
  ros2 run vlm_pipeline_live compare_backend_runs -- \
    --gt "${GT}" \
    --run "grounding_dino:${DINO}" \
    --run "owlvit:${OWL}" \
    --out "${OUT}"
else
  echo "Captures not both present yet:"
  echo "  DINO  ${DINO}  $([ -f "${DINO}" ] && echo OK || echo missing)"
  echo "  OWL   ${OWL}   $([ -f "${OWL}" ] && echo OK || echo missing)"
  echo "  GT    ${GT}    $([ -f "${GT}" ] && echo OK || echo missing)"
fi
