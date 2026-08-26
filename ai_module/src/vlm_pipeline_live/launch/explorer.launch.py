from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
  auto_start_arg = DeclareLaunchArgument(
    "auto_start",
    default_value="true",
    description="Start coverage exploration automatically after pose is available",
  )
  num_viewpoints_arg = DeclareLaunchArgument(
    "num_viewpoints",
    default_value="6",
    description="Max 360° snaps including the start pose",
  )

  explorer_node = Node(
    package="vlm_pipeline_live",
    executable="explorer_node",
    name="vlm_live_explorer",
    output="screen",
    parameters=[{
      "auto_start": LaunchConfiguration("auto_start"),
      "num_viewpoints": ParameterValue(
        LaunchConfiguration("num_viewpoints"), value_type=int
      ),
      "include_start_pose": True,
      "min_viewpoint_spacing_m": 3.0,
      "max_plan_radius_m": 25.0,
      "auto_num_viewpoints": True,
      "grid_resolution_m": 0.25,
      "free_clearance_m": 0.30,
      "wall_inset_m": 0.40,
      "bootstrap_min_points": 8000,
      "bootstrap_min_wait_sec": 2.0,
      "bootstrap_timeout_sec": 20.0,
      "position_tolerance_m": 2.0,
      "heading_tolerance_rad": 0.5,
      "skip_if_within_m": 1.5,
      "waypoint_timeout_sec": 90.0,
      "waypoint_republish_sec": 2.0,
      "min_settle_before_reached_sec": 1.0,
      "settle_sec": 4.0,
      "require_scan_stable": False,
      "detect_at_each_stop": True,
      "detect_timeout_sec": 360.0,
      "pause_after_detect_sec": 0.5,
      "detect_if_within_m": 2.5,
      "stuck_window_sec": 3.5,
      "max_stuck_recoveries": 4,
      "stuck_backup_m": 2.5,
      "stuck_backup_wait_sec": 5.0,
      "nudge_distance_m": 0.8,
      "export_pipeline_a": True,
      "shutdown_on_complete": False,
    }],
  )

  return LaunchDescription([
    auto_start_arg,
    num_viewpoints_arg,
    explorer_node,
  ])
