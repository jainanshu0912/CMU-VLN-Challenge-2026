"""Eval entrypoint: same sequential pipeline as vlm_sequential.launch.py."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
  sequential = os.path.join(
    get_package_share_directory("vlm_sequential"),
    "launch",
    "vlm_sequential.launch.py",
  )
  return LaunchDescription([
    IncludeLaunchDescription(PythonLaunchDescriptionSource(sequential)),
  ])
