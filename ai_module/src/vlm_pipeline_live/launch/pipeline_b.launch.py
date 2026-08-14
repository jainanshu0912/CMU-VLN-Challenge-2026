"""Launch Pipeline B explorer + live detector + scene graph on GPU."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEFAULT_INDOOR_PROMPT = (
  "chair . sofa . table . pillow . book . lamp . tv . monitor . plant . stool . "
  "bed . desk . cabinet . shelf . bottle . cup . bowl . window . door . picture . "
  "clock . keyboard . mouse . laptop . trash can . box . basket . curtain . mirror . "
  "rug . cushion . vase . candle . remote . phone . plate . jar . bin . ottoman"
)


def generate_launch_description():
  box_threshold_arg = DeclareLaunchArgument("box_threshold", default_value="0.28")
  text_threshold_arg = DeclareLaunchArgument("text_threshold", default_value="0.25")
  force_cpu_arg = DeclareLaunchArgument("force_cpu", default_value="false")
  device_arg = DeclareLaunchArgument("device", default_value="")

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
      "force_cpu": LaunchConfiguration("force_cpu"),
      "device": LaunchConfiguration("device"),
      "auto_run_on_exploration_complete": True,
      "shutdown_on_complete": False,
      "detection_prompt": DEFAULT_INDOOR_PROMPT,
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
      "scene_name": "live_scene",
      "near_distance_m": 1.5,
      "save_graph": True,
      "graph_output_path": "/tmp/vlm_live_scene_graph.json",
      "auto_run_on_detection_complete": True,
      "shutdown_on_complete": False,
    }],
  )

  return LaunchDescription([
    box_threshold_arg,
    text_threshold_arg,
    force_cpu_arg,
    device_arg,
    explorer_node,
    detector_node,
    scene_graph_node,
  ])
