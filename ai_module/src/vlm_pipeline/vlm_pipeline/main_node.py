"""ROS2 node for the zero-shot VLM find/count pipeline."""

import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker

from vlm_pipeline.count_pipeline import CountPipeline
from vlm_pipeline.graph_search import GraphSearchMatcher
from vlm_pipeline.query_parser import QueryParser
from vlm_pipeline.question_classifier import QuestionClassifier, QuestionType
from vlm_pipeline.scene_loader import SceneData, SceneLoader, SceneObject
from vlm_pipeline.vlm_backends import create_backend, VlmBackend

# Challenge ROS interface (matches dummy_vlm + README).
CHALLENGE_QUESTION_TOPIC = "/challenge_question"
STATE_ESTIMATION_TOPIC = "/state_estimation"
WAYPOINT_TOPIC = "/way_point_with_heading"
OBJECT_MARKER_TOPIC = "/selected_object_marker"
NUMERICAL_RESPONSE_TOPIC = "/numerical_response"

DEFAULT_VLA3D_ROOT = os.path.expanduser("~/vla3d_data/Unity")
ROS_QOS_DEPTH = 10
DUPLICATE_QUESTION_WINDOW_SEC = 2.0

# BEST_EFFORT sub receives from both BEST_EFFORT and RELIABLE publishers.
QUESTION_QOS = QoSProfile(
  depth=ROS_QOS_DEPTH,
  reliability=ReliabilityPolicy.BEST_EFFORT,
)
OUTPUT_QOS = QoSProfile(
  depth=ROS_QOS_DEPTH,
  reliability=ReliabilityPolicy.RELIABLE,
)


def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
  half_yaw = yaw / 2.0
  return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


class VlmPipelineNode(Node):
  """Subscribe to challenge questions and publish find/count responses."""

  def __init__(self) -> None:
    super().__init__("vlm_pipeline")

    self.declare_parameter("scene_name", "chinese_room")
    self.declare_parameter("vla3d_data_root", DEFAULT_VLA3D_ROOT)
    self.declare_parameter("vlm_backend", "ollama")
    self.declare_parameter("vlm_model", "")
    self.declare_parameter("use_llm_parser", False)
    self.declare_parameter("waypoint_standoff_m", 0.8)

    self.scene_name = self.get_parameter("scene_name").get_parameter_value().string_value
    data_root = self.get_parameter("vla3d_data_root").get_parameter_value().string_value
    backend_name = self.get_parameter("vlm_backend").get_parameter_value().string_value
    model_override = self.get_parameter("vlm_model").get_parameter_value().string_value
    model = model_override if model_override else None
    use_llm_parser = self.get_parameter("use_llm_parser").get_parameter_value().bool_value
    self._waypoint_standoff_m = (
      self.get_parameter("waypoint_standoff_m").get_parameter_value().double_value
    )

    self.scene_loader = SceneLoader(data_root)
    self.scene_data: SceneData = self.scene_loader.load(self.scene_name)
    self.question_classifier = QuestionClassifier()

    if use_llm_parser:
      self.vlm_backend: VlmBackend | None = create_backend(backend_name, model=model)
    else:
      self.vlm_backend = None

    self.query_parser = QueryParser(backend=self.vlm_backend, use_llm=use_llm_parser)
    self.graph_matcher = GraphSearchMatcher()
    self.count_pipeline = CountPipeline()

    self.vehicle_x = 0.0
    self.vehicle_y = 0.0
    self._pose_received = False
    self._last_marker_object: SceneObject | None = None
    self._last_question_text = ""
    self._last_question_time = 0.0

    self._setup_ros_interfaces()

    parser_mode = (
      f"llm ({self.vlm_backend.provider})"
      if use_llm_parser and self.vlm_backend is not None and self.vlm_backend.is_available()
      else "rule_based (no API key needed)"
    )
    self.get_logger().info(
      f"VLM pipeline ready | scene={self.scene_name} "
      f"| objects={len(self.scene_data.objects)} "
      f"| regions={len(self.scene_data.regions)} "
      f"| data_root={data_root} "
      f"| parser={parser_mode}"
    )
    self._log_ros_interface()

  def _setup_ros_interfaces(self) -> None:
    self.create_subscription(
      Odometry,
      STATE_ESTIMATION_TOPIC,
      self._pose_callback,
      qos_profile_sensor_data,
    )
    self.create_subscription(
      String,
      CHALLENGE_QUESTION_TOPIC,
      self._question_callback,
      QUESTION_QOS,
    )

    self.waypoint_pub = self.create_publisher(Pose2D, WAYPOINT_TOPIC, OUTPUT_QOS)
    self.object_marker_pub = self.create_publisher(Marker, OBJECT_MARKER_TOPIC, OUTPUT_QOS)
    self.numerical_answer_pub = self.create_publisher(Int32, NUMERICAL_RESPONSE_TOPIC, OUTPUT_QOS)

  def _log_ros_interface(self) -> None:
    self.get_logger().info("ROS subscriptions:")
    self.get_logger().info(f"  {STATE_ESTIMATION_TOPIC} (nav_msgs/Odometry)")
    self.get_logger().info(f"  {CHALLENGE_QUESTION_TOPIC} (std_msgs/String, BEST_EFFORT)")
    self.get_logger().info("ROS publications:")
    self.get_logger().info(f"  {OBJECT_MARKER_TOPIC} (visualization_msgs/Marker) — find")
    self.get_logger().info(f"  {WAYPOINT_TOPIC} (geometry_msgs/Pose2D) — find")
    self.get_logger().info(f"  {NUMERICAL_RESPONSE_TOPIC} (std_msgs/Int32) — count")
    self.get_logger().info("Awaiting question on /challenge_question ...")

  def _pose_callback(self, msg: Odometry) -> None:
    self.vehicle_x = msg.pose.pose.position.x
    self.vehicle_y = msg.pose.pose.position.y
    if not self._pose_received:
      self._pose_received = True
      self.get_logger().info(
        f"Receiving {STATE_ESTIMATION_TOPIC} "
        f"(robot at {self.vehicle_x:.2f}, {self.vehicle_y:.2f})"
      )

  def _question_callback(self, msg: String) -> None:
    text = msg.data.strip()
    if not text:
      self.get_logger().warn(f"Empty message on {CHALLENGE_QUESTION_TOPIC}")
      return

    now = time.monotonic()
    if (
      text == self._last_question_text
      and (now - self._last_question_time) < DUPLICATE_QUESTION_WINDOW_SEC
    ):
      self.get_logger().warn(
        f"Ignoring duplicate question on {CHALLENGE_QUESTION_TOPIC} "
        f"(within {DUPLICATE_QUESTION_WINDOW_SEC:.0f}s)"
      )
      return

    self._last_question_text = text
    self._last_question_time = now

    preview = text if len(text) <= 100 else f"{text[:97]}..."
    self.get_logger().info(f"Received question on {CHALLENGE_QUESTION_TOPIC}: {preview}")
    try:
      self._process_question(text)
    except Exception as exc:
      self.get_logger().error(f"Pipeline failed: {exc}", exc_info=True)

  def _publish_object_marker(self, obj: SceneObject) -> None:
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = self.get_clock().now().to_msg()
    marker.ns = obj.raw_label
    marker.id = int(obj.object_id)
    marker.action = Marker.ADD
    marker.type = Marker.CUBE
    marker.pose.position.x = obj.cx
    marker.pose.position.y = obj.cy
    marker.pose.position.z = obj.cz
    qx, qy, qz, qw = _yaw_to_quaternion(obj.heading)
    marker.pose.orientation.x = qx
    marker.pose.orientation.y = qy
    marker.pose.orientation.z = qz
    marker.pose.orientation.w = qw
    marker.scale.x = obj.x_length
    marker.scale.y = obj.y_length
    marker.scale.z = obj.z_length
    marker.color.a = 0.5
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    self.object_marker_pub.publish(marker)
    self.get_logger().info(f"Published {OBJECT_MARKER_TOPIC} id={obj.object_id} ns={obj.raw_label}")

  def _delete_object_marker(self, obj: SceneObject) -> None:
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = self.get_clock().now().to_msg()
    marker.ns = obj.raw_label
    marker.id = int(obj.object_id)
    marker.action = Marker.DELETE
    marker.type = Marker.CUBE
    self.object_marker_pub.publish(marker)

  def _nav_waypoint_xy(self, obj: SceneObject) -> tuple[float, float]:
    """Pick a standoff point toward the robot so the goal is on traversable area."""
    dx = self.vehicle_x - obj.cx
    dy = self.vehicle_y - obj.cy
    dist = math.hypot(dx, dy)
    if dist < 0.05:
      return obj.cx, obj.cy

    standoff = max(0.0, self._waypoint_standoff_m)
    if dist <= standoff:
      return self.vehicle_x, self.vehicle_y

    scale = standoff / dist
    return obj.cx + dx * scale, obj.cy + dy * scale

  def _publish_waypoint(self, obj: SceneObject) -> None:
    waypoint = Pose2D()
    waypoint.x, waypoint.y = self._nav_waypoint_xy(obj)
    waypoint.theta = obj.heading
    self.waypoint_pub.publish(waypoint)
    self.get_logger().info(
      f"Published {WAYPOINT_TOPIC} "
      f"({waypoint.x:.2f}, {waypoint.y:.2f}, θ={waypoint.theta:.2f}) "
      f"[object center ({obj.cx:.2f}, {obj.cy:.2f}), standoff={self._waypoint_standoff_m:.1f}m]"
    )

  def _publish_numerical_answer(self, value: int) -> None:
    self.numerical_answer_pub.publish(Int32(data=value))
    self.get_logger().info(f"Published {NUMERICAL_RESPONSE_TOPIC} = {value}")

  def _clear_marker(self) -> None:
    if self._last_marker_object is not None:
      self._delete_object_marker(self._last_marker_object)
      self._last_marker_object = None

  def _process_question(self, question: str) -> None:
    question_type = self.question_classifier.classify(question)
    self.get_logger().info(f"Question type: {question_type.value}")

    if question_type == QuestionType.FIND:
      self._handle_find(question)
    elif question_type == QuestionType.COUNT:
      self._handle_count(question)
    else:
      self._handle_navigate_stub()

  def _handle_find(self, question: str) -> None:
    parsed = self.query_parser.parse(question, QuestionType.FIND)
    self.get_logger().info(f"Parsed query ({parsed.source}): {parsed.to_dict()}")

    match = self.graph_matcher.find(self.scene_data, parsed)
    if match is None:
      self.get_logger().warn("No matching object found — no ROS output published")
      return

    self.get_logger().info(
      f"Matched object {match.object_id}: {match.raw_label} "
      f"at ({match.cx:.2f}, {match.cy:.2f}, {match.cz:.2f})"
    )
    self._clear_marker()
    self._publish_object_marker(match)
    self._publish_waypoint(match)
    self._last_marker_object = match

  def _handle_count(self, question: str) -> None:
    parsed = self.query_parser.parse(question, QuestionType.COUNT)
    self.get_logger().info(f"Parsed query ({parsed.source}): {parsed.to_dict()}")

    self._clear_marker()
    count = self.count_pipeline.count(self.scene_data, parsed)
    self.get_logger().info(f"Count result: {count}")
    self._publish_numerical_answer(count)

  def _handle_navigate_stub(self) -> None:
    self._clear_marker()
    self.get_logger().warn(
      "Navigate / instruction-following not implemented — "
      f"no waypoints published on {WAYPOINT_TOPIC}"
    )

  def destroy_node(self) -> None:
    self._clear_marker()
    super().destroy_node()


def main(args=None) -> None:
  rclpy.init(args=args)
  node = VlmPipelineNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()