"""Launch Pipeline B explorer + live detector forced to CPU (no GPU required)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CPU_TEST_PROMPT = "chair . sofa . table . pillow . book . lamp . plant . tv . stool"


def generate_launch_description():
  box_threshold_arg = DeclareLaunchArgument("box_threshold", default_value="0.28")
  text_threshold_arg = DeclareLaunchArgument("text_threshold", default_value="0.25")

  explorer_node = Node(
    package="vlm_pipeline_live",
    executable="explorer_node",
    name="vlm_live_explorer",
    output="screen",
    parameters=[{
      "auto_start": True,
      "shutdown_on_complete": False,
      "rotation_standoff_m": 0.8,
      "position_tolerance_m": 0.2,
      "min_settle_before_reached_sec": 1.0,
      "require_scan_stable": False,
      "scan_settle_sec": 6.0,
      "heading_wait_timeout_sec": 120.0,
      "waypoint_republish_sec": 2.0,
    }],
  )

  detector_node = Node(
    package="vlm_pipeline_live",
    executable="live_detector_node",
    name="vlm_live_detector",
    output="screen",
    parameters=[{
      "force_cpu": True,
      "auto_run_on_exploration_complete": True,
      "shutdown_on_complete": False,
      "detection_prompt": CPU_TEST_PROMPT,
      "box_threshold": LaunchConfiguration("box_threshold"),
      "text_threshold": LaunchConfiguration("text_threshold"),
      "model_config_path": "/home/docker/models/GroundingDINO_SwinT_OGC.py",
      "model_checkpoint_path": "/home/docker/models/groundingdino_swint_ogc.pth",
    }],
  )

  return LaunchDescription([
    box_threshold_arg,
    text_threshold_arg,
    explorer_node,
    detector_node,
  ])
