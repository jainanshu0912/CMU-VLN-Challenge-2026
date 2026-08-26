"""Launch Pipeline B coverage explorer + live detector + scene graph on GPU."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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
  box_threshold_arg = DeclareLaunchArgument("box_threshold", default_value="0.35")
  text_threshold_arg = DeclareLaunchArgument("text_threshold", default_value="0.25")
  device_arg = DeclareLaunchArgument(
    "device",
    default_value="cuda",
    description='CUDA device, e.g. "cuda" or "cuda:0"',
  )
  scene_type_arg = DeclareLaunchArgument("scene_type", default_value="office")
  scene_name_arg = DeclareLaunchArgument("scene_name", default_value="live_scene")
  num_viewpoints_arg = DeclareLaunchArgument("num_viewpoints", default_value="6")
  save_snapshots_arg = DeclareLaunchArgument("save_snapshots", default_value="true")
  snapshot_dir_arg = DeclareLaunchArgument(
    "snapshot_dir",
    default_value="/tmp/vlm_live_snapshots",
  )
  export_root_arg = DeclareLaunchArgument(
    "pipeline_a_export_root",
    default_value="/tmp/vla3d_live",
  )

  explorer_node = Node(
    package="vlm_pipeline_live",
    executable="explorer_node",
    name="vlm_live_explorer",
    output="screen",
    parameters=[{
      "auto_start": True,
      "shutdown_on_complete": False,
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
      "waypoint_republish_sec": 4.0,
      "waypoint_no_republish_within_m": 2.2,
      "settle_sec": 4.0,
      "require_scan_stable": False,
      "detect_at_each_stop": True,
      "detect_timeout_sec": 360.0,
      "pause_after_detect_sec": 0.5,
      "detect_if_within_m": 2.5,
      "stuck_window_sec": 3.5,
      "stuck_move_m": 0.15,
      "stuck_grace_sec": 3.0,
      "max_stuck_recoveries": 4,
      "stuck_backup_m": 2.5,
      "stuck_backup_wait_sec": 5.0,
      "nudge_distance_m": 1.5,
      "scene_name": LaunchConfiguration("scene_name"),
      "graph_output_dir": "/tmp/vlm_live_captures",
      "pipeline_a_export_root": LaunchConfiguration("pipeline_a_export_root"),
      "export_pipeline_a": True,
    }],
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
      "scene_type": LaunchConfiguration("scene_type"),
      "auto_run_on_exploration_complete": False,
      "allow_repeat_detection": True,
      "accumulate_detections": True,
      "save_snapshots": LaunchConfiguration("save_snapshots"),
      "snapshot_dir": LaunchConfiguration("snapshot_dir"),
      "shutdown_on_complete": False,
      "box_threshold": LaunchConfiguration("box_threshold"),
      "text_threshold": LaunchConfiguration("text_threshold"),
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
      "graph_output_path": "",
      "graph_output_dir": "/tmp/vlm_live_captures",
      "unique_graph_output": True,
      "auto_run_on_detection_complete": True,
      "shutdown_on_complete": False,
    }],
  )

  return LaunchDescription([
    detector_backend_arg,
    yolo_model_arg,
    box_threshold_arg,
    text_threshold_arg,
    device_arg,
    scene_type_arg,
    scene_name_arg,
    num_viewpoints_arg,
    save_snapshots_arg,
    snapshot_dir_arg,
    export_root_arg,
    explorer_node,
    detector_node,
    scene_graph_node,
  ])
