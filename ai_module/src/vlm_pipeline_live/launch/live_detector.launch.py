from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
  config_arg = DeclareLaunchArgument(
    "model_config_path",
    default_value="/home/docker/models/GroundingDINO_SwinT_OGC.py",
    description="GroundingDINO model config path",
  )
  checkpoint_arg = DeclareLaunchArgument(
    "model_checkpoint_path",
    default_value="/home/docker/models/groundingdino_swint_ogc.pth",
    description="GroundingDINO checkpoint path",
  )
  box_threshold_arg = DeclareLaunchArgument("box_threshold", default_value="0.3")
  text_threshold_arg = DeclareLaunchArgument("text_threshold", default_value="0.25")
  force_cpu_arg = DeclareLaunchArgument("force_cpu", default_value="false")

  detector_node = Node(
    package="vlm_pipeline_live",
    executable="live_detector_node",
    name="vlm_live_detector",
    output="screen",
    parameters=[{
      "model_config_path": LaunchConfiguration("model_config_path"),
      "model_checkpoint_path": LaunchConfiguration("model_checkpoint_path"),
      "box_threshold": LaunchConfiguration("box_threshold"),
      "text_threshold": LaunchConfiguration("text_threshold"),
      "force_cpu": LaunchConfiguration("force_cpu"),
      "auto_run_on_exploration_complete": True,
      "shutdown_on_complete": False,
    }],
  )

  return LaunchDescription([
    config_arg,
    checkpoint_arg,
    box_threshold_arg,
    text_threshold_arg,
    force_cpu_arg,
    detector_node,
  ])
