from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
  scene_name_arg = DeclareLaunchArgument(
    "scene_name",
    default_value="chinese_room",
    description="Unity scene name under vla3d_data_root/<scene_name>/",
  )
  vla3d_data_root_arg = DeclareLaunchArgument(
    "vla3d_data_root",
    default_value="/home/docker/vla3d_data/Unity",
    description="Path to VLA-3D Unity scenes (object_result.csv + scene_graph.json)",
  )
  vlm_backend_arg = DeclareLaunchArgument(
    "vlm_backend",
    default_value="ollama",
    description="LLM backend for optional parsing: ollama, gemini, gpt-4o, claude",
  )
  use_llm_parser_arg = DeclareLaunchArgument(
    "use_llm_parser",
    default_value="false",
    description="Use LLM for query parsing; false uses rule-based parser (no API key)",
  )
  vlm_model_arg = DeclareLaunchArgument(
    "vlm_model",
    default_value="",
    description="Optional model override for the LLM backend (empty uses backend default)",
  )

  vlm_pipeline_node = Node(
    package="vlm_pipeline",
    executable="vlm_pipeline_node",
    name="vlm_pipeline",
    output="screen",
    parameters=[{
      "scene_name": LaunchConfiguration("scene_name"),
      "vla3d_data_root": LaunchConfiguration("vla3d_data_root"),
      "vlm_backend": LaunchConfiguration("vlm_backend"),
      "use_llm_parser": PythonExpression([
        "'", LaunchConfiguration("use_llm_parser"), "' == 'true'",
      ]),
      "vlm_model": LaunchConfiguration("vlm_model"),
      "use_sim_time": False,
    }],
  )

  return LaunchDescription([
    scene_name_arg,
    vla3d_data_root_arg,
    vlm_backend_arg,
    use_llm_parser_arg,
    vlm_model_arg,
    vlm_pipeline_node,
  ])
