#!/usr/bin/env bash
# Hard-stop all sim / autonomy orphans inside iros2026_system.
set -euo pipefail

CONTAINER="${SIM_CONTAINER:-iros2026_system}"

docker exec "$CONTAINER" bash -c '
PIDS=$(ps -eo pid,cmd | awk "/localPlanner|pathFollower|terrainAnalysis|waypointConverter|visualizationTools|sensorScanGeneration|sim_image_repub|joy_node|vehicleSimulator|static_transform_publisher|default_server_endpoint|Model.x86_64|system_simulation|\/rviz2/ && !/awk/ {print \$1}")
if [ -n "${PIDS:-}" ]; then
  echo "Killing: $PIDS"
  for p in $PIDS; do kill -9 "$p" 2>/dev/null || true; done
else
  echo "No sim processes found"
fi
sleep 2
ps -ef | grep -E "localPlanner|vehicleSimulator|Model.x86|default_server|rviz2|system_simulation" | grep -v grep || echo "all_clear"
python3 -c "import socket;s=socket.socket();s.bind((\"0.0.0.0\",10000));print(\"port10000 FREE\");s.close()"
source /opt/ros/jazzy/setup.bash
ros2 daemon stop >/dev/null 2>&1 || true
sleep 1
ros2 daemon start >/dev/null 2>&1 || true
'
echo "Stop done."
