"""Publish challenge questions from questions.json for a given scene."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, String

CHALLENGE_QUESTION_TOPIC = "/challenge_question"
STATE_ESTIMATION_TOPIC = "/state_estimation"
WAYPOINT_TOPIC = "/way_point_with_heading"
NUMERICAL_RESPONSE_TOPIC = "/numerical_response"
ROS_QOS_DEPTH = 10

QUESTION_QOS = QoSProfile(
  depth=ROS_QOS_DEPTH,
  reliability=ReliabilityPolicy.BEST_EFFORT,
)
OUTPUT_QOS = QoSProfile(
  depth=ROS_QOS_DEPTH,
  reliability=ReliabilityPolicy.RELIABLE,
)

QUESTION_CATEGORIES = ("numerical", "object_reference", "instruction_following")


def resolve_questions_path(explicit: str) -> Path:
  """Locate questions.json from an explicit path or common install layouts."""
  if explicit:
    path = Path(explicit).expanduser()
    if path.is_file():
      return path
    raise FileNotFoundError(f"questions_path does not exist: {path}")

  env_path = os.environ.get("CMU_VLN_QUESTIONS_PATH", "").strip()
  if env_path:
    path = Path(env_path).expanduser()
    if path.is_file():
      return path

  for candidate in (
    Path("/home/docker/questions/questions.json"),
    Path.home() / "CMU-VLN-Challenge-2026/questions/questions.json",
    Path("/home/anshu/CMU-VLN-Challenge-2026/questions/questions.json"),
  ):
    if candidate.is_file():
      return candidate

  for root in Path(__file__).resolve().parents:
    candidate = root / "questions" / "questions.json"
    if candidate.is_file():
      return candidate

  raise FileNotFoundError(
    "Could not find questions.json. Set parameter questions_path or "
    "environment variable CMU_VLN_QUESTIONS_PATH."
  )


def load_questions_for_scene(
  questions_path: Path,
  scene_name: str,
  category: str,
) -> list[str]:
  data = json.loads(questions_path.read_text(encoding="utf-8"))
  for entry in data:
    if entry.get("scene") != scene_name:
      continue
    questions_dict = entry.get("questions", {})
    if category != "all":
      if category not in QUESTION_CATEGORIES:
        raise ValueError(
          f"Invalid question_type '{category}'. "
          f"Use: all, {', '.join(QUESTION_CATEGORIES)}"
        )
      return list(questions_dict.get(category, []))

    ordered: list[str] = []
    for key in QUESTION_CATEGORIES:
      ordered.extend(questions_dict.get(key, []))
    return ordered

  raise ValueError(f"Scene '{scene_name}' not found in {questions_path}")


def classify_question(question: str) -> str:
  """Match vlm_pipeline.question_classifier routing."""
  text = question.strip().lower()
  if text.startswith("how many") or text.startswith("count"):
    return "count"
  if text.startswith("find") or text.startswith("the "):
    return "find"
  return "navigate"


class PublishQuestionsNode(Node):
  """Publish training questions for one scene, then exit."""

  def __init__(self) -> None:
    super().__init__("publish_questions")

    self.declare_parameter("scene_name", "")
    self.declare_parameter("questions_path", "")
    self.declare_parameter("question_type", "all")
    self.declare_parameter("wait_for_subscriber", True)
    self.declare_parameter("wait_timeout", 30.0)
    self.declare_parameter("waypoint_reach_distance", 1.0)
    self.declare_parameter("waypoint_wait_timeout", 120.0)
    self.declare_parameter("response_wait_timeout", 30.0)
    self.declare_parameter("post_arrival_delay", 2.0)
    self.declare_parameter("post_count_delay", 3.0)
    self.declare_parameter("post_navigate_delay", 15.0)
    self.declare_parameter("no_progress_timeout", 45.0)

    scene_name = self.get_parameter("scene_name").get_parameter_value().string_value.strip()
    if not scene_name:
      raise ValueError("scene_name parameter is required")

    questions_path = resolve_questions_path(
      self.get_parameter("questions_path").get_parameter_value().string_value.strip()
    )
    category = self.get_parameter("question_type").get_parameter_value().string_value.strip()
    self._wait_for_subscriber = (
      self.get_parameter("wait_for_subscriber").get_parameter_value().bool_value
    )
    self._wait_timeout = self.get_parameter("wait_timeout").get_parameter_value().double_value
    self._waypoint_reach_distance = (
      self.get_parameter("waypoint_reach_distance").get_parameter_value().double_value
    )
    self._waypoint_wait_timeout = (
      self.get_parameter("waypoint_wait_timeout").get_parameter_value().double_value
    )
    self._response_wait_timeout = (
      self.get_parameter("response_wait_timeout").get_parameter_value().double_value
    )
    self._post_arrival_delay = (
      self.get_parameter("post_arrival_delay").get_parameter_value().double_value
    )
    self._post_count_delay = (
      self.get_parameter("post_count_delay").get_parameter_value().double_value
    )
    self._post_navigate_delay = (
      self.get_parameter("post_navigate_delay").get_parameter_value().double_value
    )
    self._no_progress_timeout = (
      self.get_parameter("no_progress_timeout").get_parameter_value().double_value
    )

    self._questions = load_questions_for_scene(questions_path, scene_name, category)
    if not self._questions:
      raise ValueError(
        f"No questions for scene={scene_name}, question_type={category} in {questions_path}"
      )

    self.get_logger().info(
      f"Loaded {len(self._questions)} question(s) for scene={scene_name} "
      f"(type={category}) from {questions_path}"
    )

    self.vehicle_x = 0.0
    self.vehicle_y = 0.0
    self._pose_received = False

    self._target_x: float | None = None
    self._target_y: float | None = None
    self._target_theta: float = 0.0
    self._waypoint_received = False
    self._count_received = False
    self._nav_waypoint_count = 0
    self._active_question_id = 0
    self._waypoint_question_id = 0

    self._publisher = self.create_publisher(String, CHALLENGE_QUESTION_TOPIC, QUESTION_QOS)
    self.create_subscription(
      Odometry,
      STATE_ESTIMATION_TOPIC,
      self._pose_callback,
      qos_profile_sensor_data,
    )
    self.create_subscription(Pose2D, WAYPOINT_TOPIC, self._waypoint_callback, OUTPUT_QOS)
    self.create_subscription(Int32, NUMERICAL_RESPONSE_TOPIC, self._count_callback, OUTPUT_QOS)

  def run(self) -> None:
    if self._wait_for_subscriber:
      self._wait_for_subscriber_loop()

    total = len(self._questions)
    for index, question in enumerate(self._questions):
      self._active_question_id += 1
      self._reset_response_state()
      self._waypoint_question_id = self._active_question_id

      message = String()
      message.data = question
      # Publish once per question so vlm_pipeline is not triggered twice.
      self._publisher.publish(message)

      self.get_logger().info(f"Published [{index + 1}/{total}]: {question}")
      self._wait_for_question_completion(question)

  def _reset_response_state(self) -> None:
    self._target_x = None
    self._target_y = None
    self._target_theta = 0.0
    self._waypoint_received = False
    self._count_received = False
    self._nav_waypoint_count = 0
    self._waypoint_question_id = 0

  def _pose_callback(self, msg: Odometry) -> None:
    self.vehicle_x = msg.pose.pose.position.x
    self.vehicle_y = msg.pose.pose.position.y
    if not self._pose_received:
      self._pose_received = True
      self.get_logger().info(
        f"Receiving {STATE_ESTIMATION_TOPIC} "
        f"(robot at {self.vehicle_x:.2f}, {self.vehicle_y:.2f})"
      )

  def _waypoint_callback(self, msg: Pose2D) -> None:
    if self._waypoint_question_id != self._active_question_id:
      return

    self._target_x = msg.x
    self._target_y = msg.y
    self._target_theta = msg.theta
    self._waypoint_received = True
    self._nav_waypoint_count += 1
    self.get_logger().info(
      f"Waypoint for Q{self._active_question_id}: ({msg.x:.2f}, {msg.y:.2f}) "
      f"[seq={self._nav_waypoint_count}]"
    )

  def _count_callback(self, msg: Int32) -> None:
    self._count_received = True
    self.get_logger().info(f"Count response received: {msg.data}")

  def _distance_to_target(self) -> float:
    if self._target_x is None or self._target_y is None:
      return float("inf")
    return math.hypot(self.vehicle_x - self._target_x, self.vehicle_y - self._target_y)

  def _spin_until(self, deadline: float) -> None:
    while time.monotonic() < deadline and rclpy.ok():
      rclpy.spin_once(self, timeout_sec=0.1)

  def _wait_for_waypoint_response(self) -> bool:
    deadline = time.monotonic() + self._response_wait_timeout
    self.get_logger().info(
      f"Waiting for waypoint on {WAYPOINT_TOPIC} "
      f"(timeout={self._response_wait_timeout:.0f}s)..."
    )
    while time.monotonic() < deadline and rclpy.ok():
      rclpy.spin_once(self, timeout_sec=0.1)
      if self._waypoint_received:
        return True
    self.get_logger().warn("No waypoint published for this question.")
    return False

  def _wait_until_at_waypoint(self) -> bool:
    if self._target_x is None or self._target_y is None:
      return False

    if not self._pose_received:
      self.get_logger().warn(
        f"No pose on {STATE_ESTIMATION_TOPIC} yet — arrival checks may be wrong."
      )

    distance = self._distance_to_target()
    if distance <= self._waypoint_reach_distance:
      self.get_logger().info(
        f"Already within {self._waypoint_reach_distance:.1f}m of waypoint "
        f"(distance={distance:.2f}m)."
      )
      self._finish_arrival_pause()
      return True

    deadline = time.monotonic() + self._waypoint_wait_timeout
    start_distance = distance
    best_distance = distance
    last_log = time.monotonic()
    last_progress = time.monotonic()

    self.get_logger().info(
      f"Waiting for robot to reach ({self._target_x:.2f}, {self._target_y:.2f}) "
      f"from ({self.vehicle_x:.2f}, {self.vehicle_y:.2f}), "
      f"distance={distance:.2f}m, "
      f"reach={self._waypoint_reach_distance:.1f}m "
      f"(timeout={self._waypoint_wait_timeout:.0f}s)..."
    )

    while time.monotonic() < deadline and rclpy.ok():
      rclpy.spin_once(self, timeout_sec=0.1)
      distance = self._distance_to_target()

      if distance <= self._waypoint_reach_distance:
        self.get_logger().info(f"Robot reached waypoint (distance={distance:.2f}m).")
        self._finish_arrival_pause()
        return True

      if distance < best_distance - 0.02:
        best_distance = distance
        last_progress = time.monotonic()

      now = time.monotonic()
      if now - last_log >= 5.0:
        self.get_logger().info(
          f"Still en route: robot=({self.vehicle_x:.2f}, {self.vehicle_y:.2f}), "
          f"distance={distance:.2f}m, best={best_distance:.2f}m"
        )
        last_log = now

      if now - last_progress >= self._no_progress_timeout:
        self.get_logger().warn(
          f"Navigation stalled for {self._no_progress_timeout:.0f}s "
          f"(start={start_distance:.2f}m, best={best_distance:.2f}m, "
          f"current={distance:.2f}m). Continuing to next question."
        )
        return False

    self.get_logger().warn(
      f"Timed out waiting to reach waypoint "
      f"(distance={self._distance_to_target():.2f}m). Continuing to next question."
    )
    return False

  def _finish_arrival_pause(self) -> None:
    if self._post_arrival_delay > 0:
      self.get_logger().info(
        f"Pausing {self._post_arrival_delay:.1f}s before next question."
      )
      self._spin_until(time.monotonic() + self._post_arrival_delay)

  def _wait_for_count_response(self) -> None:
    deadline = time.monotonic() + self._response_wait_timeout
    self.get_logger().info(
      f"Waiting for count on {NUMERICAL_RESPONSE_TOPIC} "
      f"(timeout={self._response_wait_timeout:.0f}s)..."
    )
    while time.monotonic() < deadline and rclpy.ok():
      rclpy.spin_once(self, timeout_sec=0.1)
      if self._count_received:
        if self._post_count_delay > 0:
          self._spin_until(time.monotonic() + self._post_count_delay)
        return
    self.get_logger().warn("No count response received; using post_count_delay fallback.")
    if self._post_count_delay > 0:
      self._spin_until(time.monotonic() + self._post_count_delay)

  def _wait_for_navigation_complete(self) -> None:
    """Wait for waypoint sequence: reach each waypoint, then idle before next."""
    if not self._wait_for_waypoint_response():
      self._spin_until(time.monotonic() + self._post_navigate_delay)
      return

    while rclpy.ok():
      if not self._wait_until_at_waypoint():
        break

      idle_deadline = time.monotonic() + 5.0
      waypoint_count_before = self._nav_waypoint_count
      while time.monotonic() < idle_deadline and rclpy.ok():
        rclpy.spin_once(self, timeout_sec=0.1)
        if self._nav_waypoint_count > waypoint_count_before:
          break
      else:
        break

    if self._post_arrival_delay > 0:
      self._spin_until(time.monotonic() + self._post_arrival_delay)

  def _wait_for_question_completion(self, question: str) -> None:
    qtype = classify_question(question)
    if qtype == "find":
      if self._wait_for_waypoint_response():
        self._wait_until_at_waypoint()
    elif qtype == "count":
      self._wait_for_count_response()
    else:
      self.get_logger().info(
        f"Navigate question — waiting for waypoint sequence "
        f"(fallback delay={self._post_navigate_delay:.0f}s)..."
      )
      self._wait_for_navigation_complete()

  def _wait_for_subscriber_loop(self) -> None:
    self.get_logger().info(
      f"Waiting for subscriber on {CHALLENGE_QUESTION_TOPIC} "
      f"(timeout={self._wait_timeout}s)..."
    )
    deadline = time.monotonic() + self._wait_timeout
    while time.monotonic() < deadline:
      rclpy.spin_once(self, timeout_sec=0.1)
      if self._publisher.get_subscription_count() > 0:
        self.get_logger().info("Subscriber detected.")
        return
    self.get_logger().warn("No subscriber detected; publishing anyway.")


def main(args=None) -> None:
  rclpy.init(args=args)
  node = PublishQuestionsNode()
  try:
    node.run()
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()
