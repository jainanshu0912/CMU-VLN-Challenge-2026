"""Launch Pipeline C (SORT3D-style) reasoning node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description() -> LaunchDescription:
  pkg_share = get_package_share_directory("vlm_pipeline_sort3d")
  default_config = os.path.join(pkg_share, "config", "pipeline_c.yaml")

  return LaunchDescription([
    DeclareLaunchArgument("scene_name", default_value="chinese_room"),
    DeclareLaunchArgument("config", default_value=default_config),
    Node(
      package="vlm_pipeline_sort3d",
      executable="sort3d_node",
      name="sort3d_node",
      output="screen",
      parameters=[
        LaunchConfiguration("config"),
        {"scene_name": LaunchConfiguration("scene_name")},
      ],
    ),
  ])
