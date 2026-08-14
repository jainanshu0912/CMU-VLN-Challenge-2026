#!/usr/bin/env bash
# Reliable sim start: libdl + NVIDIA + endpoint BEFORE Unity + RViz.
# Usage (on host):
#   ./docker/sim_stop.sh
#   ./docker/sim_start.sh
set -euo pipefail

CONTAINER="${SIM_CONTAINER:-iros2026_system}"
MESH_HOST="${MESH_HOST:-$HOME/CMU-VLN-Challenge-2026/autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/mesh/unity/environment}"

xhost + >/dev/null 2>&1 || true
if [ -d "$MESH_HOST" ]; then
  chmod -R a+rwX "$MESH_HOST" 2>/dev/null || true
  chmod o+w "$MESH_HOST" 2>/dev/null || true
fi

docker exec -u 0 "$CONTAINER" bash -c '
ln -sfn /usr/lib/x86_64-linux-gnu/libdl.so.2 /usr/lib/x86_64-linux-gnu/libdl.so
mkdir -p /home/docker/.config/unity3d/UnityRobotics/cmu_vla_challenge_unity
: > /home/docker/.config/unity3d/UnityRobotics/cmu_vla_challenge_unity/Player.log
'

# Ensure nothing leftover
"$(dirname "$0")/sim_stop.sh"

echo "Starting ROS launch (endpoint first)..."
docker exec -d "$CONTAINER" bash -lc '
export DISPLAY="${DISPLAY:-:0}"
export QT_X11_NO_MITSHM=1
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=all
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/fastdds_udp.xml}"
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
cd /home/docker/autonomy_stack_mecanum_wheel_platform
source ./install/setup.bash
nohup ros2 launch vehicle_simulator system_simulation.launch > /tmp/sim_launch.log 2>&1 &
'

echo "Waiting for ROS-TCP :10000..."
ok=0
for i in $(seq 1 30); do
  if docker exec "$CONTAINER" python3 -c 'import socket;s=socket.socket();s.settimeout(0.5);s.connect(("127.0.0.1",10000))' 2>/dev/null; then
    echo "endpoint OK (${i}s)"
    ok=1
    break
  fi
  sleep 1
done
if [ "$ok" -ne 1 ]; then
  echo "ERROR: endpoint did not bind on :10000" >&2
  docker exec "$CONTAINER" tail -40 /tmp/sim_launch.log 2>/dev/null || true
  exit 1
fi

echo "Starting Unity..."
docker exec -d "$CONTAINER" bash -lc '
export DISPLAY="${DISPLAY:-:0}"
export QT_X11_NO_MITSHM=1
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=all
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/fastdds_udp.xml}"
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
cd /home/docker/autonomy_stack_mecanum_wheel_platform
nohup ./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 > /tmp/unity.log 2>&1 &
'

echo "Waiting for /registered_scan..."
for i in $(seq 1 40); do
  hz=$(docker exec "$CONTAINER" bash -lc '
    source /opt/ros/jazzy/setup.bash
    source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash
    timeout 3 ros2 topic hz /registered_scan 2>&1 | grep -o "average rate: [0-9.]*" | tail -1
  ' 2>/dev/null || true)
  if [ -n "$hz" ]; then
    echo "scan OK: $hz"
    break
  fi
  sleep 1
  if [ "$i" -eq 40 ]; then
    echo "WARN: /registered_scan not up yet; check Player.log" >&2
    docker exec "$CONTAINER" bash -lc 'grep -E "Renderer:|DllNotFound|Unauthorized|HeaderMsg" /home/docker/.config/unity3d/UnityRobotics/cmu_vla_challenge_unity/Player.log | tail -20'
  fi
done

echo "Starting RViz..."
docker exec -d "$CONTAINER" bash -lc '
export DISPLAY="${DISPLAY:-:0}"
export QT_X11_NO_MITSHM=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/fastdds_udp.xml}"
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
cd /home/docker/autonomy_stack_mecanum_wheel_platform
source ./install/setup.bash
nohup ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz > /tmp/rviz.log 2>&1 &
'

echo
echo "Done. In RViz: Fixed Frame = map, enable PointCloud2 /registered_scan."
echo "Quick check:"
echo "  docker exec -it $CONTAINER bash -lc 'source /opt/ros/jazzy/setup.bash; source ~/autonomy_stack_mecanum_wheel_platform/install/setup.bash; timeout 5 ros2 topic hz /registered_scan'"
