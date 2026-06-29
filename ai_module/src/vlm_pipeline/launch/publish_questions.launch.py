from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
  scene_name_arg = DeclareLaunchArgument(
    "scene_name",
    default_value="chinese_room",
    description="Unity scene name (must match vlm_pipeline scene_name)",
  )
  questions_path_arg = DeclareLaunchArgument(
    "questions_path",
    default_value="",
    description="Path to questions.json (empty = auto-detect)",
  )
  question_type_arg = DeclareLaunchArgument(
    "question_type",
    default_value="all",
    description="all | numerical | object_reference | instruction_following",
  )
  waypoint_reach_distance_arg = DeclareLaunchArgument(
    "waypoint_reach_distance",
    default_value="1.0",
    description="Distance (m) to waypoint before publishing next question",
  )
  waypoint_wait_timeout_arg = DeclareLaunchArgument(
    "waypoint_wait_timeout",
    default_value="120.0",
    description="Max seconds to wait for robot to reach waypoint",
  )

  publish_questions_node = Node(
    package="vlm_pipeline",
    executable="publish_questions",
    name="publish_questions",
    output="screen",
    parameters=[{
      "scene_name": LaunchConfiguration("scene_name"),
      "questions_path": LaunchConfiguration("questions_path"),
      "question_type": LaunchConfiguration("question_type"),
      "wait_for_subscriber": True,
      "wait_timeout": 30.0,
      "waypoint_reach_distance": ParameterValue(
        LaunchConfiguration("waypoint_reach_distance"), value_type=float
      ),
      "waypoint_wait_timeout": ParameterValue(
        LaunchConfiguration("waypoint_wait_timeout"), value_type=float
      ),
      "response_wait_timeout": 30.0,
      "post_arrival_delay": 2.0,
      "post_count_delay": 3.0,
      "post_navigate_delay": 15.0,
      "no_progress_timeout": 45.0,
    }],
  )

  return LaunchDescription([
    scene_name_arg,
    questions_path_arg,
    question_type_arg,
    waypoint_reach_distance_arg,
    waypoint_wait_timeout_arg,
    publish_questions_node,
  ])
