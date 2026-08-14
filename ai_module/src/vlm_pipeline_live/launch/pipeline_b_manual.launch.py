"""Manual teleop mapping on GPU: live detector + scene graph (no auto exploration)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Full indoor prompt (GPU can afford the longer caption).
DEFAULT_INDOOR_PROMPT = (
  "chair . sofa . table . pillow . book . lamp . tv . monitor . plant . stool . "
  "bed . desk . cabinet . shelf . bottle . cup . bowl . window . door . picture . "
  "clock . keyboard . mouse . laptop . trash can . box . basket . curtain . mirror . "
  "rug . cushion . vase . candle . remote . phone . plate . jar . bin . ottoman"
)


def generate_launch_description():
  box_threshold_arg = DeclareLaunchArgument("box_threshold", default_value="0.28")
  text_threshold_arg = DeclareLaunchArgument("text_threshold", default_value="0.25")
  force_cpu_arg = DeclareLaunchArgument(
    "force_cpu",
    default_value="false",
    description="Set true to force CPU even if CUDA is available",
  )
  device_arg = DeclareLaunchArgument(
    "device",
    default_value="",
    description='Empty = auto (cuda if available). Examples: "cuda", "cuda:0", "cpu"',
  )
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
      "force_cpu": LaunchConfiguration("force_cpu"),
      "device": LaunchConfiguration("device"),
      "auto_run_on_exploration_complete": False,
      "allow_repeat_detection": True,
      "accumulate_detections": True,
      "save_snapshots": LaunchConfiguration("save_snapshots"),
      "snapshot_dir": LaunchConfiguration("snapshot_dir"),
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
    save_snapshots_arg,
    snapshot_dir_arg,
    detector_node,
    scene_graph_node,
  ])
