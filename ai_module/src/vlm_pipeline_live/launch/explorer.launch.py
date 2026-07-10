from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
  auto_start_arg = DeclareLaunchArgument(
    "auto_start",
    default_value="true",
    description="Start exploration automatically after pose is available",
  )

  explorer_node = Node(
    package="vlm_pipeline_live",
    executable="explorer_node",
    name="vlm_live_explorer",
    output="screen",
    parameters=[{
      "auto_start": LaunchConfiguration("auto_start"),
      "headings_deg": [0.0, 90.0, 180.0, 270.0],
      "rotation_standoff_m": 0.8,
      "position_tolerance_m": 0.2,
      "min_settle_before_reached_sec": 1.0,
      "heading_tolerance_rad": 0.35,
      "heading_wait_timeout_sec": 120.0,
      "waypoint_republish_sec": 2.0,
      "scan_stable_window_sec": 5.0,
      "scan_change_threshold": 0.05,
      "scan_stable_timeout_sec": 30.0,
      "require_scan_stable": False,
      "scan_settle_sec": 6.0,
      "pause_after_view_sec": 1.0,
      "shutdown_on_complete": True,
    }],
  )

  return LaunchDescription([
    auto_start_arg,
    explorer_node,
  ])
