"""Manual teleop mapping on GPU: live detector + scene graph (no auto exploration)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
  scene_name_arg = DeclareLaunchArgument(
    "scene_name",
    default_value="office_2",
    description="Name used in scene graph JSON / archive (match Unity world when possible)",
  )
  scene_type_arg = DeclareLaunchArgument(
    "scene_type",
    default_value="office",
    description="Prompt preset: Unity scene name (office_2, hotel_room_1, …) or office|hotel|livingroom|home|cultural|indoor",
  )
  detection_prompt_arg = DeclareLaunchArgument(
    "detection_prompt",
    default_value="",
    description="Optional dotted caption override; empty → scene_type preset",
  )
  detector_backend_arg = DeclareLaunchArgument(
    "detector_backend",
    default_value="grounding_dino",
    description="2D open-vocab backend: grounding_dino | yolo_world | yoloe | owlvit",
  )
  yolo_model_arg = DeclareLaunchArgument(
    "yolo_model",
    default_value="",
    description=(
      "Weights / HF id for yolo_world, yoloe, or owlvit. "
      "Empty → yolov8s-worldv2.pt, yoloe-11s-seg.pt, or google/owlv2-base-patch16."
    ),
  )
  box_threshold_arg = DeclareLaunchArgument(
    "box_threshold",
    default_value="0.35",
    description="2D box confidence floor (higher → fewer extras)",
  )
  text_threshold_arg = DeclareLaunchArgument("text_threshold", default_value="0.25")
  device_arg = DeclareLaunchArgument(
    "device",
    default_value="cuda",
    description='CUDA device, e.g. "cuda" or "cuda:0"',
  )
  save_snapshots_arg = DeclareLaunchArgument("save_snapshots", default_value="true")
  snapshot_dir_arg = DeclareLaunchArgument(
    "snapshot_dir",
    default_value="/tmp/vlm_live_snapshots",
  )
  graph_output_arg = DeclareLaunchArgument(
    "graph_output_path",
    default_value="",
    description="Optional fixed *.json path (will be timestamp-suffixed). Empty → graph_output_dir",
  )
  graph_output_dir_arg = DeclareLaunchArgument(
    "graph_output_dir",
    default_value="/tmp/vlm_live_captures",
    description="Root for unique captures: <dir>/<scene_name>/<timestamp>/scene_graph.json",
  )
  unique_graph_arg = DeclareLaunchArgument(
    "unique_graph_output",
    default_value="true",
    description="If true, never overwrite prior scene graphs",
  )
  nms_distance_arg = DeclareLaunchArgument(
    "nms_distance_m",
    default_value="0.5",
    description="Base 3D NMS distance; large objects use a bigger class-aware radius",
  )
  gemini_verify_arg = DeclareLaunchArgument(
    "gemini_verify",
    default_value="false",
    description="If true, Gemini Flash verifies/relabels/drops 2D boxes before LiDAR fusion",
  )
  gemini_model_arg = DeclareLaunchArgument(
    "gemini_model",
    default_value="gemini-3.6-flash",
    description="Free-tier Gemini model for label verification",
  )
  gemini_api_key_arg = DeclareLaunchArgument(
    "gemini_api_key",
    default_value="",
    description="Optional; empty → GEMINI_API_KEY / GOOGLE_API_KEY env",
  )

  detector_node = Node(
    package="vlm_pipeline_live",
    executable="live_detector_node",
    name="vlm_live_detector",
    output="screen",
    parameters=[{
      "detector_backend": LaunchConfiguration("detector_backend"),
      "yolo_model": LaunchConfiguration("yolo_model"),
      "device": LaunchConfiguration("device"),
      "auto_run_on_exploration_complete": False,
      "allow_repeat_detection": True,
      "accumulate_detections": True,
      "save_snapshots": LaunchConfiguration("save_snapshots"),
      "snapshot_dir": LaunchConfiguration("snapshot_dir"),
      "shutdown_on_complete": False,
      "scene_type": LaunchConfiguration("scene_type"),
      "detection_prompt": LaunchConfiguration("detection_prompt"),
      "box_threshold": LaunchConfiguration("box_threshold"),
      "text_threshold": LaunchConfiguration("text_threshold"),
      "nms_distance_m": LaunchConfiguration("nms_distance_m"),
      "gemini_verify": LaunchConfiguration("gemini_verify"),
      "gemini_model": LaunchConfiguration("gemini_model"),
      "gemini_api_key": LaunchConfiguration("gemini_api_key"),
      "model_config_path": "/home/docker/models/GroundingDINO_SwinT_OGC.py",
      "model_checkpoint_path": "/home/docker/models/groundingdino_swint_ogc.pth",
    }],
  )

  scene_graph_node = Node(
    package="vlm_pipeline_live",
    executable="live_scene_graph_node",
    name="vlm_live_scene_graph",
    output="screen",
    parameters=[{
      "scene_name": LaunchConfiguration("scene_name"),
      "near_distance_m": 1.5,
      "save_graph": True,
      "graph_output_path": LaunchConfiguration("graph_output_path"),
      "graph_output_dir": LaunchConfiguration("graph_output_dir"),
      "unique_graph_output": LaunchConfiguration("unique_graph_output"),
      "auto_run_on_detection_complete": True,
      "shutdown_on_complete": False,
    }],
  )

  return LaunchDescription([
    scene_name_arg,
    scene_type_arg,
    detection_prompt_arg,
    detector_backend_arg,
    yolo_model_arg,
    box_threshold_arg,
    text_threshold_arg,
    device_arg,
    save_snapshots_arg,
    snapshot_dir_arg,
    graph_output_arg,
    graph_output_dir_arg,
    unique_graph_arg,
    nms_distance_arg,
    gemini_verify_arg,
    gemini_model_arg,
    gemini_api_key_arg,
    detector_node,
    scene_graph_node,
  ])
