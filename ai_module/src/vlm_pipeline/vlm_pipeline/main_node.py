"""ROS2 node for the zero-shot VLM find/count/navigate pipeline."""

import math
import os
import threading
import time
from typing import Optional

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Int32, String
from visualization_msgs.msg import Marker

from vlm_pipeline.count_pipeline import CountPipeline
from vlm_pipeline.graph_search import GraphSearchMatcher
from vlm_pipeline.navigate_parser import NavigateParser
from vlm_pipeline.navigate_pipeline import NavigatePipeline, ResolvedWaypoint, format_waypoint_summary
from vlm_pipeline.query_parser import QueryParser
from vlm_pipeline.question_classifier import QuestionClassifier, QuestionType
from vlm_pipeline.scene_loader import SceneData, SceneLoader, SceneObject
from vlm_pipeline.vlm_backends import create_backend, VlmBackend
from vlm_pipeline.waypoint_recovery import backup_waypoint

# Challenge ROS interface (matches dummy_vlm + README).
CHALLENGE_QUESTION_TOPIC = "/challenge_question"
STATE_ESTIMATION_TOPIC = "/state_estimation"
WAYPOINT_TOPIC = "/way_point_with_heading"
WAYPOINT_REACHED_TOPIC = "/way_point_reached"
OBJECT_MARKER_TOPIC = "/selected_object_marker"
NUMERICAL_RESPONSE_TOPIC = "/numerical_response"

DEFAULT_VLA3D_ROOT = os.path.expanduser("~/vla3d_data/Unity")
# dummy_vlm and waypoint_converter use create_*(..., 5) → RELIABLE KeepLast(5).
ROS_QOS_DEPTH = 5

# dummy_vlm uses create_subscription(..., 5) → RELIABLE KeepLast. Eval publishes
# to that. A BEST_EFFORT-only sub does not match a RELIABLE eval publisher.
QUESTION_QOS = QoSProfile(
  depth=ROS_QOS_DEPTH,
  reliability=ReliabilityPolicy.RELIABLE,
)
QUESTION_QOS_BEST_EFFORT = QoSProfile(
  depth=ROS_QOS_DEPTH,
  reliability=ReliabilityPolicy.BEST_EFFORT,
)
# Match vehicleSimulator /state_estimation (RELIABLE). sensor_data BEST_EFFORT
# often fails to deliver pose to this node on Jazzy + FastDDS.
POSE_QOS = QoSProfile(
  depth=ROS_QOS_DEPTH,
  reliability=ReliabilityPolicy.RELIABLE,
  durability=DurabilityPolicy.VOLATILE,
  history=HistoryPolicy.KEEP_LAST,
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

  def __init__(self, node_name: str = "vlm_pipeline") -> None:
    super().__init__(node_name)

    self.declare_parameter("scene_name", "chinese_room")
    self.declare_parameter("vla3d_data_root", DEFAULT_VLA3D_ROOT)
    self.declare_parameter("defer_scene_load", False)
    self.declare_parameter("vlm_backend", "ollama")
    self.declare_parameter("vlm_model", "")
    self.declare_parameter("use_llm_parser", False)
    # Standoff keeps goals off object footprints. Autonomy waypointXYRadius≈0.3m.
    self.declare_parameter("waypoint_standoff_m", 0.7)
    self.declare_parameter("waypoint_reach_m", 0.45)
    self.declare_parameter("waypoint_timeout_sec", 120.0)
    self.declare_parameter("waypoint_republish_sec", 2.0)
    self.declare_parameter("waypoint_reach_dwell_sec", 0.3)
    # If already this close to the object center, skip the leg (don't publish
    # a same-pose goal — autonomy treats that as zero motion / stuck).
    self.declare_parameter("waypoint_skip_if_within_m", 0.9)
    # Stop republishing inside this radius — republish resets autonomy waypointReached
    # and causes endless rotate-in-place near the goal.
    self.declare_parameter("waypoint_no_republish_within_m", 0.6)
    self.declare_parameter("stuck_window_sec", 3.5)
    self.declare_parameter("stuck_move_m", 0.15)
    self.declare_parameter("stuck_grace_sec", 3.0)
    self.declare_parameter("stuck_backup_m", 1.2)
    self.declare_parameter("stuck_backup_wait_sec", 4.0)
    self.declare_parameter("max_stuck_recoveries", 3)

    self.scene_name = self.get_parameter("scene_name").get_parameter_value().string_value
    data_root = self.get_parameter("vla3d_data_root").get_parameter_value().string_value
    defer_scene_load = self.get_parameter("defer_scene_load").get_parameter_value().bool_value
    backend_name = self.get_parameter("vlm_backend").get_parameter_value().string_value
    model_override = self.get_parameter("vlm_model").get_parameter_value().string_value
    model = model_override if model_override else None
    use_llm_parser = self.get_parameter("use_llm_parser").get_parameter_value().bool_value
    self._waypoint_standoff_m = (
      self.get_parameter("waypoint_standoff_m").get_parameter_value().double_value
    )
    self._waypoint_reach_m = (
      self.get_parameter("waypoint_reach_m").get_parameter_value().double_value
    )
    self._waypoint_timeout_sec = (
      self.get_parameter("waypoint_timeout_sec").get_parameter_value().double_value
    )
    self._waypoint_republish_sec = (
      self.get_parameter("waypoint_republish_sec").get_parameter_value().double_value
    )
    self._waypoint_reach_dwell_sec = (
      self.get_parameter("waypoint_reach_dwell_sec").get_parameter_value().double_value
    )
    self._waypoint_skip_if_within_m = (
      self.get_parameter("waypoint_skip_if_within_m").get_parameter_value().double_value
    )
    self._waypoint_no_republish_within_m = (
      self.get_parameter("waypoint_no_republish_within_m").get_parameter_value().double_value
    )
    self._stuck_window_sec = (
      self.get_parameter("stuck_window_sec").get_parameter_value().double_value
    )
    self._stuck_move_m = (
      self.get_parameter("stuck_move_m").get_parameter_value().double_value
    )
    self._stuck_grace_sec = (
      self.get_parameter("stuck_grace_sec").get_parameter_value().double_value
    )
    self._stuck_backup_m = (
      self.get_parameter("stuck_backup_m").get_parameter_value().double_value
    )
    self._stuck_backup_wait_sec = (
      self.get_parameter("stuck_backup_wait_sec").get_parameter_value().double_value
    )
    self._max_stuck_recoveries = int(self.get_parameter("max_stuck_recoveries").value)

    self.scene_loader = SceneLoader(data_root)
    self.scene_data: Optional[SceneData] = None
    if not defer_scene_load:
      self.scene_data = self.scene_loader.load(self.scene_name)
    self.question_classifier = QuestionClassifier()

    if use_llm_parser:
      self.vlm_backend: VlmBackend | None = create_backend(backend_name, model=model)
    else:
      self.vlm_backend = None

    self.query_parser = QueryParser(backend=self.vlm_backend, use_llm=use_llm_parser)
    self.graph_matcher = GraphSearchMatcher()
    self.count_pipeline = CountPipeline()
    self.navigate_parser = NavigateParser()
    self.navigate_pipeline = NavigatePipeline(
      matcher=self.graph_matcher,
      query_parser=self.query_parser,
      standoff_m=self._waypoint_standoff_m,
    )

    self.vehicle_x = 0.0
    self.vehicle_y = 0.0
    self.vehicle_yaw = 0.0
    self._pose_received = False
    self._navigating = False
    self._cancel_navigation = False
    self._queued_question: Optional[str] = None
    self._question_lock = threading.Lock()
    self._waypoint_reached_flag = False
    self._last_marker_object: SceneObject | None = None
    self._last_question_text = ""
    self._handled_questions: set[str] = set()
    self._handled_lock = threading.Lock()
    self._last_count: Optional[int] = None

    self._setup_ros_interfaces()
    self.create_timer(1.0, self._republish_scored_outputs)

    parser_mode = (
      f"llm ({self.vlm_backend.provider})"
      if use_llm_parser and self.vlm_backend is not None and self.vlm_backend.is_available()
      else "rule_based (no API key needed)"
    )
    if self.scene_data is None:
      self.get_logger().info(
        f"VLM pipeline waiting for live scene graph | scene={self.scene_name} "
        f"| data_root={data_root} | parser={parser_mode}"
      )
    else:
      self.get_logger().info(
        f"VLM pipeline ready | scene={self.scene_name} "
        f"| objects={len(self.scene_data.objects)} "
        f"| regions={len(self.scene_data.regions)} "
        f"| data_root={data_root} "
        f"| parser={parser_mode}"
      )
    self._log_ros_interface()

  def _setup_ros_interfaces(self) -> None:
    # Reentrant so pose callbacks can run while find/navigate waits for pose.
    self._cb_group = ReentrantCallbackGroup()
    self.create_subscription(
      Odometry,
      STATE_ESTIMATION_TOPIC,
      self._pose_callback,
      POSE_QOS,
      callback_group=self._cb_group,
    )
    self.create_subscription(
      String,
      CHALLENGE_QUESTION_TOPIC,
      self._question_callback,
      QUESTION_QOS,
      callback_group=self._cb_group,
    )
    self.create_subscription(
      String,
      CHALLENGE_QUESTION_TOPIC,
      self._question_callback,
      QUESTION_QOS_BEST_EFFORT,
      callback_group=self._cb_group,
    )
    self.create_subscription(
      Float32,
      WAYPOINT_REACHED_TOPIC,
      self._waypoint_reached_callback,
      OUTPUT_QOS,
      callback_group=self._cb_group,
    )

    self.waypoint_pub = self.create_publisher(Pose2D, WAYPOINT_TOPIC, OUTPUT_QOS)
    self.object_marker_pub = self.create_publisher(Marker, OBJECT_MARKER_TOPIC, OUTPUT_QOS)
    self.numerical_answer_pub = self.create_publisher(Int32, NUMERICAL_RESPONSE_TOPIC, OUTPUT_QOS)

  def load_deferred_scene(
    self,
    data_root: Optional[str] = None,
    scene_name: Optional[str] = None,
  ) -> bool:
    """Load CSV + scene graph after a live capture (used by vlm_sequential)."""
    if data_root:
      self.scene_loader = SceneLoader(data_root)
    if scene_name:
      self.scene_name = scene_name
    try:
      self.scene_data = self.scene_loader.load(self.scene_name)
    except (OSError, ValueError) as exc:
      self.get_logger().error(f"Failed to load scene '{self.scene_name}': {exc}")
      self.scene_data = None
      return False
    self.get_logger().info(
      f"Loaded scene graph | scene={self.scene_name} "
      f"| objects={len(self.scene_data.objects)} "
      f"| regions={len(self.scene_data.regions)} "
      f"| data_root={self.scene_loader.data_root}"
    )
    return True

  def _log_ros_interface(self) -> None:
    self.get_logger().info("ROS subscriptions:")
    self.get_logger().info(f"  {STATE_ESTIMATION_TOPIC} (nav_msgs/Odometry)")
    self.get_logger().info(
      f"  {CHALLENGE_QUESTION_TOPIC} (std_msgs/String, RELIABLE + BEST_EFFORT)"
    )
    self.get_logger().info(f"  {WAYPOINT_REACHED_TOPIC} (std_msgs/Float32)")
    self.get_logger().info("ROS publications:")
    self.get_logger().info(f"  {OBJECT_MARKER_TOPIC} (visualization_msgs/Marker) — find")
    self.get_logger().info(
      f"  {WAYPOINT_TOPIC} (geometry_msgs/Pose2D) — find + navigate sequence"
    )
    self.get_logger().info(f"  {NUMERICAL_RESPONSE_TOPIC} (std_msgs/Int32) — count")
    self.get_logger().info("Awaiting question on /challenge_question ...")

  def _pose_callback(self, msg: Odometry) -> None:
    self.vehicle_x = msg.pose.pose.position.x
    self.vehicle_y = msg.pose.pose.position.y
    q = msg.pose.pose.orientation
    # yaw from quaternion (z-up)
    self.vehicle_yaw = math.atan2(
      2.0 * (q.w * q.z + q.x * q.y),
      1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
    if not self._pose_received:
      self._pose_received = True
      self.get_logger().info(
        f"Receiving {STATE_ESTIMATION_TOPIC} "
        f"(robot at {self.vehicle_x:.2f}, {self.vehicle_y:.2f})"
      )

  def _waypoint_reached_callback(self, _msg: Float32) -> None:
    if self._navigating:
      self._waypoint_reached_flag = True

  def _question_callback(self, msg: String) -> None:
    text = msg.data.strip()
    if not text:
      self.get_logger().warn(f"Empty message on {CHALLENGE_QUESTION_TOPIC}")
      return

    # Eval republishes the same String at 1 Hz, and this node has two
    # subscribers (RELIABLE + BEST_EFFORT). Claim the text once so we never
    # DELETE+ADD the find cube again in the same episode.
    with self._handled_lock:
      if text in self._handled_questions:
        return
      self._handled_questions.add(text)

    # If find/count/navigate is in progress, queue the latest *different*
    # question and cancel navigate so the holder can pick it up after abort.
    if not self._question_lock.acquire(blocking=False):
      preview = text if len(text) <= 100 else f"{text[:97]}..."
      self.get_logger().warn(
        f"Preempting active question for new message on "
        f"{CHALLENGE_QUESTION_TOPIC}: {preview}"
      )
      self._queued_question = text
      self._cancel_navigation = True
      return

    try:
      self._run_question_and_drain_queue(text)
    finally:
      self._question_lock.release()

  def _run_question_and_drain_queue(self, text: str) -> None:
    """Process ``text``, then any newer question queued by preemption.

    Caller must hold ``_question_lock``.
    """
    current: Optional[str] = text
    while current is not None:
      self._last_question_text = current
      preview = current if len(current) <= 100 else f"{current[:97]}..."
      self.get_logger().info(
        f"Received question on {CHALLENGE_QUESTION_TOPIC}: {preview}"
      )
      try:
        self._process_question(current)
      except Exception as exc:
        self.get_logger().error(f"Pipeline failed: {exc}", exc_info=True)

      # Keep only the latest queued preempt (may have been overwritten mid-run).
      current = self._queued_question
      self._queued_question = None
      self._cancel_navigation = False
      if current is not None:
        self.get_logger().info(
          "Taking up queued challenge question after cancel"
        )

  def _publish_object_marker(self, obj: SceneObject, *, log: bool = True) -> None:
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = self.get_clock().now().to_msg()
    # Must match vehicle_simulator.rviz Marker display namespace filter.
    marker.ns = "my_markers"
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
    marker.lifetime.sec = 0
    marker.lifetime.nanosec = 0
    self.object_marker_pub.publish(marker)
    if log:
      self.get_logger().info(
        f"Published {OBJECT_MARKER_TOPIC} id={obj.object_id} label={obj.raw_label} ns=my_markers"
      )

  def _delete_object_marker(self, obj: SceneObject) -> None:
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = self.get_clock().now().to_msg()
    marker.ns = "my_markers"
    marker.id = int(obj.object_id)
    marker.action = Marker.DELETE
    marker.type = Marker.CUBE
    self.object_marker_pub.publish(marker)

  def _effective_standoff_m(self, obj: SceneObject) -> float:
    """Keep goals outside the object bbox (half diagonal + margin)."""
    half_extent = 0.5 * math.hypot(max(obj.x_length, 0.05), max(obj.y_length, 0.05))
    return max(self._waypoint_standoff_m, half_extent + 0.45)

  def _distance_to_object(self, obj: SceneObject) -> float:
    return math.hypot(self.vehicle_x - obj.cx, self.vehicle_y - obj.cy)

  def _nav_waypoint_xy(self, obj: SceneObject) -> tuple[float, float]:
    """Pick a standoff point toward the robot so the goal is on traversable area."""
    dx = self.vehicle_x - obj.cx
    dy = self.vehicle_y - obj.cy
    dist = math.hypot(dx, dy)
    standoff = self._effective_standoff_m(obj)
    if dist < 0.05:
      # Degenerate: step back along -x in map if we have no approach direction.
      return obj.cx - standoff, obj.cy

    # Always return a point at `standoff` from the object — never the robot pose
    # (same-pose goals make the autonomy stack stop moving).
    scale = standoff / dist
    return obj.cx + dx * scale, obj.cy + dy * scale

  def _wait_for_pose(self, timeout_sec: float = 5.0) -> bool:
    if self._pose_received:
      return True
    deadline = time.monotonic() + timeout_sec
    self.get_logger().warn(
      f"No pose yet on {STATE_ESTIMATION_TOPIC}; waiting up to {timeout_sec:.0f}s..."
    )
    # Do not nested-spin the node here (breaks SingleThreadedExecutor). With a
    # MultiThreadedExecutor + reentrant group, pose callbacks arrive while we sleep.
    while time.monotonic() < deadline and rclpy.ok():
      if self._pose_received:
        return True
      time.sleep(0.05)
    return self._pose_received

  def _publish_waypoint(self, obj: SceneObject) -> None:
    waypoint = Pose2D()
    waypoint.x, waypoint.y = self._nav_waypoint_xy(obj)
    waypoint.theta = math.atan2(obj.cy - self.vehicle_y, obj.cx - self.vehicle_x)
    self.waypoint_pub.publish(waypoint)
    self.get_logger().info(
      f"Published {WAYPOINT_TOPIC} "
      f"({waypoint.x:.2f}, {waypoint.y:.2f}, θ={waypoint.theta:.2f}) "
      f"[object center ({obj.cx:.2f}, {obj.cy:.2f}), standoff={self._waypoint_standoff_m:.1f}m]"
    )

  def _publish_numerical_answer(self, value: int, *, log: bool = True) -> None:
    self._last_count = value
    self.numerical_answer_pub.publish(Int32(data=value))
    if log:
      self.get_logger().info(f"Published {NUMERICAL_RESPONSE_TOPIC} = {value}")

  def _republish_scored_outputs(self) -> None:
    """Eval subscribers are volatile; keep the last scored output on the wire."""
    if self._last_count is not None:
      self._publish_numerical_answer(self._last_count, log=False)
    elif self._last_marker_object is not None:
      self._publish_object_marker(self._last_marker_object, log=False)

  def _clear_marker(self) -> None:
    if self._last_marker_object is not None:
      self._delete_object_marker(self._last_marker_object)
      self._last_marker_object = None

  def _process_question(self, question: str) -> None:
    if self.scene_data is None:
      self.get_logger().warn(
        "No scene graph loaded yet — buffering is handled by vlm_sequential; "
        "dropping question in standalone mode"
      )
      return

    question_type = self.question_classifier.classify(question)
    self.get_logger().info(f"Question type: {question_type.value}")

    if question_type == QuestionType.FIND:
      self._handle_find(question)
    elif question_type == QuestionType.COUNT:
      self._handle_count(question)
    else:
      self._handle_navigate(question)

  def _handle_find(self, question: str) -> None:
    if not self._wait_for_pose(timeout_sec=3.0):
      self.get_logger().error(
        f"No pose on {STATE_ESTIMATION_TOPIC} after 3s — "
        "skipping find (standoff would be computed from origin)"
      )
      return

    self._last_count = None
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
    if (
      self._last_marker_object is None
      or self._last_marker_object.object_id != match.object_id
    ):
      self._publish_object_marker(match)
      self._last_marker_object = match

    goal_x, goal_y = self._nav_waypoint_xy(match)
    theta = math.atan2(match.cy - self.vehicle_y, match.cx - self.vehicle_x)
    goal = ResolvedWaypoint(
      x=goal_x,
      y=goal_y,
      theta=theta,
      label=match.raw_label,
      object_id=str(match.object_id),
      kind="find",
    )
    dist = math.hypot(self.vehicle_x - goal.x, self.vehicle_y - goal.y)
    if dist < 0.2:
      self.get_logger().info("Find: already at standoff — marker published, not driving")
      return

    self.get_logger().info(
      f"Find: driving to ({goal.x:.2f}, {goal.y:.2f}) "
      f"from ({self.vehicle_x:.2f}, {self.vehicle_y:.2f}) "
      f"(re-publishing {WAYPOINT_TOPIC} until reached). "
      "RViz must stay in Waypoint mode."
    )
    self._navigating = True
    self._cancel_navigation = False
    try:
      self._drive_to_waypoint(goal, target_obj=match)
    finally:
      self._navigating = False

  def _handle_count(self, question: str) -> None:
    parsed = self.query_parser.parse(question, QuestionType.COUNT)
    self.get_logger().info(f"Parsed query ({parsed.source}): {parsed.to_dict()}")

    self._clear_marker()
    count = self.count_pipeline.count(self.scene_data, parsed)
    self.get_logger().info(f"Count result: {count}")
    self._publish_numerical_answer(count)

  def _handle_navigate(self, question: str) -> None:
    if not self._wait_for_pose(timeout_sec=3.0):
      self.get_logger().error(
        f"No pose on {STATE_ESTIMATION_TOPIC} after 3s — skipping navigate"
      )
      return

    self._clear_marker()
    self._last_count = None
    parsed = self.navigate_parser.parse(question)
    self.get_logger().info(f"Parsed navigate ({parsed.source}): {parsed.to_dict()}")
    if parsed.avoid_phrases:
      self.get_logger().info(
        f"Avoid constraints noted (not geometrically enforced yet): {parsed.avoid_phrases}"
      )

    waypoints = self.navigate_pipeline.resolve(
      self.scene_data,
      parsed,
      robot_x=self.vehicle_x,
      robot_y=self.vehicle_y,
    )
    if not waypoints:
      self.get_logger().warn("No navigate waypoints resolved — nothing published")
      return

    self.get_logger().info(
      f"Navigate path ({len(waypoints)} waypoints): {format_waypoint_summary(waypoints)}"
    )
    self._follow_waypoint_sequence(waypoints)

  def _follow_waypoint_sequence(self, waypoints: list[ResolvedWaypoint]) -> None:
    self._navigating = True
    self._cancel_navigation = False
    try:
      for index, wp in enumerate(waypoints):
        if self._cancel_navigation:
          self.get_logger().warn(
            f"Navigate cancelled before leg {index + 1}/{len(waypoints)} "
            f"(new challenge question)"
          )
          return

        obj = None
        if wp.object_id:
          obj = self.scene_data.get_object(str(wp.object_id))

        # Already beside the object → advance (publishing current pose freezes the stack).
        if obj is not None:
          dist_obj = self._distance_to_object(obj)
          skip_r = max(self._waypoint_skip_if_within_m, self._effective_standoff_m(obj))
          if dist_obj <= skip_r:
            self.get_logger().info(
              f"Navigate leg {index + 1}/{len(waypoints)}: already near "
              f"{wp.label} (dist={dist_obj:.2f}m ≤ {skip_r:.2f}m) — skipping"
            )
            continue

        goal = self._refresh_navigate_goal(wp)
        # Same-pose / tiny goals also freeze waypointConverter — skip.
        if math.hypot(goal.x - self.vehicle_x, goal.y - self.vehicle_y) < 0.2:
          self.get_logger().info(
            f"Navigate leg {index + 1}/{len(waypoints)}: goal already at robot — skipping"
          )
          continue

        self.get_logger().info(
          f"Navigate leg {index + 1}/{len(waypoints)}: {goal.kind} {goal.label} "
          f"→ ({goal.x:.2f}, {goal.y:.2f}) "
          f"robot=({self.vehicle_x:.2f}, {self.vehicle_y:.2f})"
        )
        if not self._drive_to_waypoint(goal, target_obj=obj):
          dist = math.hypot(self.vehicle_x - goal.x, self.vehicle_y - goal.y)
          reason = (
            "new challenge question"
            if self._cancel_navigation
            else "timeout or shutdown"
          )
          self.get_logger().warn(
            f"Navigate stopped at leg {index + 1}/{len(waypoints)} "
            f"({reason}); dist_to_goal={dist:.2f}m "
            f"robot=({self.vehicle_x:.2f}, {self.vehicle_y:.2f}) "
            f"goal=({goal.x:.2f}, {goal.y:.2f})"
          )
          return
      self.get_logger().info("Navigate path complete")
    finally:
      self._navigating = False

  def _refresh_navigate_goal(self, wp: ResolvedWaypoint) -> ResolvedWaypoint:
    """Update object goals using current pose (avoids stale chained standoffs)."""
    if not wp.object_id:
      theta = math.atan2(wp.y - self.vehicle_y, wp.x - self.vehicle_x)
      return ResolvedWaypoint(
        x=wp.x,
        y=wp.y,
        theta=theta,
        label=wp.label,
        object_id=wp.object_id,
        kind=wp.kind,
      )

    obj = self.scene_data.get_object(str(wp.object_id))
    if obj is None:
      return wp

    x, y = self._nav_waypoint_xy(obj)
    theta = math.atan2(obj.cy - self.vehicle_y, obj.cx - self.vehicle_x)
    return ResolvedWaypoint(
      x=x,
      y=y,
      theta=theta,
      label=wp.label or obj.raw_label,
      object_id=str(obj.object_id),
      kind=wp.kind,
    )

  def _drive_to_waypoint(
    self,
    wp: ResolvedWaypoint,
    target_obj: Optional[SceneObject] = None,
  ) -> bool:
    """Publish a waypoint and wait until autonomy reports reached / we are close."""
    self._waypoint_reached_flag = False
    # Approach heading toward the goal (not a forced look-at that causes spinning).
    theta = math.atan2(wp.y - self.vehicle_y, wp.x - self.vehicle_x)
    self._publish_xy_waypoint(wp.x, wp.y, theta)
    deadline = time.monotonic() + self._waypoint_timeout_sec
    last_publish = time.monotonic()
    started = time.monotonic()
    last_motion_x = self.vehicle_x
    last_motion_y = self.vehicle_y
    last_motion_time = started
    recoveries = 0
    reach = max(0.35, self._waypoint_reach_m)
    no_repub = max(reach, self._waypoint_no_republish_within_m)
    dwell = max(0.0, self._waypoint_reach_dwell_sec)
    inside_since: Optional[float] = None

    while time.monotonic() < deadline and rclpy.ok():
      if self._cancel_navigation:
        self.get_logger().warn(
          f"Drive cancelled en route to ({wp.x:.2f}, {wp.y:.2f}) "
          f"(new challenge question)"
        )
        return False

      time.sleep(0.05)
      now = time.monotonic()
      dist = math.hypot(self.vehicle_x - wp.x, self.vehicle_y - wp.y)
      dist_obj = (
        self._distance_to_object(target_obj) if target_obj is not None else None
      )

      moved = math.hypot(self.vehicle_x - last_motion_x, self.vehicle_y - last_motion_y)
      if moved >= self._stuck_move_m:
        last_motion_x = self.vehicle_x
        last_motion_y = self.vehicle_y
        last_motion_time = now

      # Autonomy stack publishes this when within waypointXYRadius (~0.3m).
      if self._waypoint_reached_flag and now - started > 0.4:
        self.get_logger().info(
          f"Autonomy /way_point_reached near ({wp.x:.2f}, {wp.y:.2f}) "
          f"robot=({self.vehicle_x:.2f}, {self.vehicle_y:.2f}) dist={dist:.2f}m"
        )
        return True

      close_enough = dist < reach
      if dist_obj is not None:
        close_enough = close_enough or dist_obj <= self._waypoint_skip_if_within_m

      if close_enough and now - started > 0.4:
        if inside_since is None:
          inside_since = now
        elif now - inside_since >= dwell:
          self.get_logger().info(
            f"Reached waypoint ({wp.x:.2f}, {wp.y:.2f}) "
            f"robot=({self.vehicle_x:.2f}, {self.vehicle_y:.2f}) dist={dist:.2f}m"
            + (f" dist_obj={dist_obj:.2f}m" if dist_obj is not None else "")
          )
          return True
      else:
        inside_since = None

      stuck = (
        now - started >= self._stuck_grace_sec
        and (now - last_motion_time) >= self._stuck_window_sec
        and dist > no_repub
      )
      if stuck and recoveries < self._max_stuck_recoveries:
        recoveries += 1
        if self._unstick_with_backup(wp, recovery_index=recoveries, deadline=deadline):
          return True
        last_publish = time.monotonic()
        last_motion_x = self.vehicle_x
        last_motion_y = self.vehicle_y
        last_motion_time = last_publish
        self._waypoint_reached_flag = False
        continue

      # Critical: do NOT republish when close — that resets waypointReached and
      # makes the robot rotate in place forever near the goal.
      if dist > no_repub and now - last_publish >= self._waypoint_republish_sec:
        theta = math.atan2(wp.y - self.vehicle_y, wp.x - self.vehicle_x)
        self._publish_xy_waypoint(wp.x, wp.y, theta, log=False)
        last_publish = now

    return False

  def _unstick_with_backup(
    self,
    wp: ResolvedWaypoint,
    *,
    recovery_index: int,
    deadline: float,
  ) -> bool:
    """Drive a short backup waypoint, then republish the original target."""
    bx, by, btheta = backup_waypoint(
      self.vehicle_x,
      self.vehicle_y,
      self.vehicle_yaw,
      wp.x,
      wp.y,
      backup_m=self._stuck_backup_m,
      recovery_index=recovery_index,
    )
    self.get_logger().warn(
      f"Stuck at ({self.vehicle_x:.2f}, {self.vehicle_y:.2f}) "
      f"goal=({wp.x:.2f}, {wp.y:.2f}) — backing up to "
      f"({bx:.2f}, {by:.2f}) recovery {recovery_index}/{self._max_stuck_recoveries}"
    )
    self._publish_xy_waypoint(bx, by, btheta)
    backup_deadline = min(deadline, time.monotonic() + self._stuck_backup_wait_sec)
    start_x, start_y = self.vehicle_x, self.vehicle_y
    while time.monotonic() < backup_deadline and rclpy.ok():
      if self._cancel_navigation:
        return False
      time.sleep(0.05)
      if math.hypot(self.vehicle_x - wp.x, self.vehicle_y - wp.y) <= max(
        0.35, self._waypoint_reach_m
      ):
        return True
      if math.hypot(self.vehicle_x - start_x, self.vehicle_y - start_y) >= self._stuck_move_m:
        break

    theta = math.atan2(wp.y - self.vehicle_y, wp.x - self.vehicle_x)
    self.get_logger().info(
      f"Retrying original target ({wp.x:.2f}, {wp.y:.2f}) after backup"
    )
    self._publish_xy_waypoint(wp.x, wp.y, theta)
    return False

  def _publish_xy_waypoint(
    self,
    x: float,
    y: float,
    theta: float,
    log: bool = True,
  ) -> None:
    waypoint = Pose2D()
    waypoint.x = x
    waypoint.y = y
    waypoint.theta = theta
    self.waypoint_pub.publish(waypoint)
    if log:
      self.get_logger().info(
        f"Published {WAYPOINT_TOPIC} ({x:.2f}, {y:.2f}, θ={theta:.2f})"
      )

  def destroy_node(self) -> None:
    self._clear_marker()
    super().destroy_node()


def main(args=None) -> None:
  rclpy.init(args=args)
  node = VlmPipelineNode()
  executor = MultiThreadedExecutor(num_threads=4)
  executor.add_node(node)
  try:
    executor.spin()
  except KeyboardInterrupt:
    pass
  finally:
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()