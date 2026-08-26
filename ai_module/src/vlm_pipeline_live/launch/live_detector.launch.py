"""Launch live detector only (GPU). Prefer pipeline_b_manual.launch.py for full flow."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
  detector_backend_arg = DeclareLaunchArgument(
    "detector_backend",
    default_value="grounding_dino",
    description="2D open-vocab backend: grounding_dino | yolo_world | yoloe | owlvit",
  )
  yolo_model_arg = DeclareLaunchArgument(
    "yolo_model",
    default_value="",
    description="Weights / HF id for yolo_world, yoloe, or owlvit (empty → backend default)",
  )
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
  box_threshold_arg = DeclareLaunchArgument("box_threshold", default_value="0.35")
  text_threshold_arg = DeclareLaunchArgument("text_threshold", default_value="0.25")
  device_arg = DeclareLaunchArgument(
    "device",
    default_value="cuda",
    description='CUDA device, e.g. "cuda" or "cuda:0"',
  )
  scene_type_arg = DeclareLaunchArgument("scene_type", default_value="office")

  detector_node = Node(
    package="vlm_pipeline_live",
    executable="live_detector_node",
    name="vlm_live_detector",
    output="screen",
    parameters=[{
      "detector_backend": LaunchConfiguration("detector_backend"),
      "yolo_model": LaunchConfiguration("yolo_model"),
      "model_config_path": LaunchConfiguration("model_config_path"),
      "model_checkpoint_path": LaunchConfiguration("model_checkpoint_path"),
      "box_threshold": LaunchConfiguration("box_threshold"),
      "text_threshold": LaunchConfiguration("text_threshold"),
      "device": LaunchConfiguration("device"),
      "scene_type": LaunchConfiguration("scene_type"),
      "auto_run_on_exploration_complete": True,
      "shutdown_on_complete": False,
    }],
  )

  return LaunchDescription([
    detector_backend_arg,
    yolo_model_arg,
    config_arg,
    checkpoint_arg,
    box_threshold_arg,
    text_threshold_arg,
    device_arg,
    scene_type_arg,
    detector_node,
  ])
