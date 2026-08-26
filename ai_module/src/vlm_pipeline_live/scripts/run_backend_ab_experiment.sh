#!/usr/bin/env bash
# Three-way Pipeline B backend capture helper (manual teleop).
#
# Usage (inside iros2026_ai_module, with sim + autonomy already up):
#   bash /home/docker/ai_module/src/vlm_pipeline_live/scripts/run_backend_ab_experiment.sh
#
# For each backend you will:
#   1) launch detector+graph
#   2) teleop to the SAME viewpoints
#   3) trigger /vlm_live/run_detection a few times
#   4) Ctrl-C the launch when done (graph already saved under that scene_name)
#
# Then this script runs compare_backend_runs vs Unity office_2 GT.

set -euo pipefail

ROOT="${AI_MODULE_ROOT:-/home/docker/ai_module}"
GT="${GT_SCENE_GRAPH:-/home/docker/vla3d_data/Unity/office_2/office_2_scene_graph.json}"
OUT_DIR="${COMPARE_OUT_DIR:-/tmp/vlm_backend_compare}"
CAPTURE_ROOT="${CAPTURE_ROOT:-/tmp/vlm_live_captures}"

source /opt/ros/jazzy/setup.bash
# Prefer overlay if present
if [[ -f "${ROOT}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/install/setup.bash"
fi

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo " Pipeline B backend A/B/C experiment (office_2)"
echo " GT: ${GT}"
echo "============================================================"
echo
echo "Run these THREE launches one at a time (stop each with Ctrl-C when done)."
echo "Use the SAME teleop viewpoints + same number of /vlm_live/run_detection triggers."
echo

cat <<EOF
# --- 1) GroundingDINO only ---
ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py \\
  scene_name:=office_2_dino \\
  scene_type:=office \\
  detector_backend:=grounding_dino \\
  gemini_verify:=false \\
  save_snapshots:=true \\
  graph_output_dir:=${CAPTURE_ROOT}

# Teleop → trigger a few times → Ctrl-C

# --- 2) GroundingDINO + Gemini verify ---
ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py \\
  scene_name:=office_2_dino_gemini \\
  scene_type:=office \\
  detector_backend:=grounding_dino \\
  gemini_verify:=true \\
  gemini_model:=gemini-3.6-flash \\
  save_snapshots:=true \\
  graph_output_dir:=${CAPTURE_ROOT}

# Same teleop tour → Ctrl-C

# --- 3) YOLOE (YOLO11 open-vocab) ---
ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py \\
  scene_name:=office_2_yoloe \\
  scene_type:=office \\
  detector_backend:=yoloe \\
  gemini_verify:=false \\
  save_snapshots:=true \\
  graph_output_dir:=${CAPTURE_ROOT}

# Same teleop tour → Ctrl-C
EOF

echo
echo "When all three latest_scene_graph.json files exist, press Enter to compare…"
read -r _

DINO="${CAPTURE_ROOT}/office_2_dino/latest_scene_graph.json"
DGEM="${CAPTURE_ROOT}/office_2_dino_gemini/latest_scene_graph.json"
YOLO="${CAPTURE_ROOT}/office_2_yoloe/latest_scene_graph.json"

for p in "${DINO}" "${DGEM}" "${YOLO}"; do
  if [[ ! -f "${p}" ]]; then
    echo "Missing: ${p}"
    echo "Finish that backend capture first."
    exit 1
  fi
done

OUT_JSON="${OUT_DIR}/office_2_backend_compare.json"
ros2 run vlm_pipeline_live compare_backend_runs -- \
  --gt "${GT}" \
  --run "grounding_dino:${DINO}" \
  --run "dino_gemini:${DGEM}" \
  --run "yoloe:${YOLO}" \
  --out "${OUT_JSON}"

echo
echo "Done. Summary JSON → ${OUT_JSON}"
echo "Individual captures:"
echo "  ${DINO}"
echo "  ${DGEM}"
echo "  ${YOLO}"
