"""Launch live scene-graph builder (subscribes to detection outputs)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
  scene_name_arg = DeclareLaunchArgument("scene_name", default_value="live_scene")
  near_arg = DeclareLaunchArgument("near_distance_m", default_value="1.5")
  save_arg = DeclareLaunchArgument("save_graph", default_value="true")
  path_arg = DeclareLaunchArgument(
    "graph_output_path",
    default_value="/tmp/vlm_live_scene_graph.json",
  )

  node = Node(
    package="vlm_pipeline_live",
    executable="live_scene_graph_node",
    name="vlm_live_scene_graph",
    output="screen",
    parameters=[{
      "scene_name": LaunchConfiguration("scene_name"),
      "near_distance_m": LaunchConfiguration("near_distance_m"),
      "save_graph": LaunchConfiguration("save_graph"),
      "graph_output_path": LaunchConfiguration("graph_output_path"),
      "auto_run_on_detection_complete": True,
      "shutdown_on_complete": False,
    }],
  )

  return LaunchDescription([
    scene_name_arg,
    near_arg,
    save_arg,
    path_arg,
    node,
  ])
