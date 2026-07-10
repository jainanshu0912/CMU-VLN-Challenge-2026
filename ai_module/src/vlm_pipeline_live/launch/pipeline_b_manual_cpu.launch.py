"""Manual teleop mapping: live detector only (no auto exploration)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CPU_TEST_PROMPT = "chair . sofa . table . pillow . book . lamp . plant . tv . stool"


def generate_launch_description():
  box_threshold_arg = DeclareLaunchArgument("box_threshold", default_value="0.28")
  text_threshold_arg = DeclareLaunchArgument("text_threshold", default_value="0.25")
  save_snapshots_arg = DeclareLaunchArgument("save_snapshots", default_value="true")
  snapshot_dir_arg = DeclareLaunchArgument(
    "snapshot_dir",
    default_value="/tmp/vlm_live_snapshots",
  )

  detector_node = Node(
    package="vlm_pipeline_live",
    executable="live_detector_node",
    name="vlm_live_detector",
    output="screen",
    parameters=[{
      "force_cpu": True,
      "auto_run_on_exploration_complete": False,
      "allow_repeat_detection": True,
      "accumulate_detections": True,
      "save_snapshots": LaunchConfiguration("save_snapshots"),
      "snapshot_dir": LaunchConfiguration("snapshot_dir"),
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
    save_snapshots_arg,
    snapshot_dir_arg,
    detector_node,
  ])
