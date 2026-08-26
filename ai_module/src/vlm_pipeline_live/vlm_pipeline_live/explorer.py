"""Coverage explorer: plan 5–6 room standpoints from /registered_scan and detect at each."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
  DurabilityPolicy,
  HistoryPolicy,
  QoSProfile,
  ReliabilityPolicy,
  qos_profile_sensor_data,
)

from geometry_msgs.msg import Point, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from vlm_pipeline_live.lidar_camera_fusion import pointcloud2_to_xyz
from vlm_pipeline_live.scan_stability import ScanStabilityMonitor, point_cloud_size
from vlm_pipeline_live.viewpoint_planner import (
  Viewpoint,
  ViewpointPlannerConfig,
  order_nearest_neighbor,
  plan_coverage_viewpoints,
  plan_next_coverage_viewpoint,
  suggested_viewpoint_count,
)
from vlm_pipeline.waypoint_recovery import backup_waypoint
from vlm_pipeline_live.process_handshake import (
  DETECTION_COMPLETE as HS_DETECTION_COMPLETE,
  EXPLORATION_COMPLETE as HS_EXPLORATION_COMPLETE,
  RUN_DETECTION as HS_RUN_DETECTION,
  SCENE_GRAPH_COMPLETE as HS_SCENE_GRAPH_COMPLETE,
  bump,
  read_token,
  write_token,
)
from vlm_pipeline_live.write_object_list_from_scene_graph import export_scene_folder

STATE_ESTIMATION_TOPIC = "/state_estimation"
REGISTERED_SCAN_TOPIC = "/registered_scan"
WAYPOINT_TOPIC = "/way_point_with_heading"
EXPLORATION_COMPLETE_TOPIC = "/vlm_live/exploration_complete"
RUN_DETECTION_TOPIC = "/vlm_live/run_detection"
DETECTION_COMPLETE_TOPIC = "/vlm_live/detection_complete"
SCENE_GRAPH_COMPLETE_TOPIC = "/vlm_live/scene_graph_complete"
VIEWPOINT_MARKERS_TOPIC = "/vlm_live/explorer_viewpoints"

OUTPUT_QOS = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
LIVE_QOS = qos_profile_sensor_data
# Latch so vlm_sequential still sees completion if it subscribes late.
COMPLETE_QOS = QoSProfile(
  depth=1,
  reliability=ReliabilityPolicy.RELIABLE,
  durability=DurabilityPolicy.TRANSIENT_LOCAL,
  history=HistoryPolicy.KEEP_LAST,
)


def yaw_from_odometry(msg: Odometry) -> float:
  q = msg.pose.pose.orientation
  sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
  cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
  return math.atan2(sin_yaw, cos_yaw)


def normalize_angle(angle: float) -> float:
  while angle > math.pi:
    angle -= 2.0 * math.pi
  while angle < -math.pi:
    angle += 2.0 * math.pi
  return angle


def angular_error(current_yaw: float, target_yaw: float) -> float:
  return abs(normalize_angle(target_yaw - current_yaw))


class ExplorerNode(Node):
  """Drive a short coverage tour and trigger a 360° detection at each stop."""

  def __init__(self) -> None:
    super().__init__("vlm_live_explorer")

    self.declare_parameter("auto_start", True)
    self.declare_parameter("shutdown_on_complete", False)
    self.declare_parameter("num_viewpoints", 6)
    self.declare_parameter("auto_num_viewpoints", True)
    self.declare_parameter("include_start_pose", True)
    self.declare_parameter("min_viewpoint_spacing_m", 3.0)
    self.declare_parameter("grid_resolution_m", 0.25)
    self.declare_parameter("free_clearance_m", 0.30)
    self.declare_parameter("wall_inset_m", 0.40)
    self.declare_parameter("z_min_m", 0.15)
    self.declare_parameter("z_max_m", 1.8)
    self.declare_parameter("occupied_count_min", 3)
    self.declare_parameter("max_plan_radius_m", 25.0)
    self.declare_parameter("candidate_stride_m", 0.5)
    self.declare_parameter("bootstrap_min_points", 8000)
    self.declare_parameter("bootstrap_min_wait_sec", 2.0)
    self.declare_parameter("bootstrap_timeout_sec", 20.0)
    self.declare_parameter("position_tolerance_m", 2.0)
    self.declare_parameter("heading_tolerance_rad", 0.5)
    self.declare_parameter("skip_if_within_m", 1.5)
    self.declare_parameter("waypoint_timeout_sec", 90.0)
    self.declare_parameter("waypoint_republish_sec", 4.0)
    self.declare_parameter("waypoint_no_republish_within_m", 2.2)
    self.declare_parameter("min_settle_before_reached_sec", 1.0)
    self.declare_parameter("settle_sec", 4.0)
    self.declare_parameter("require_scan_stable", False)
    self.declare_parameter("scan_stable_window_sec", 5.0)
    self.declare_parameter("scan_change_threshold", 0.05)
    self.declare_parameter("scan_stable_timeout_sec", 30.0)
    self.declare_parameter("detect_at_each_stop", True)
    self.declare_parameter("detect_timeout_sec", 360.0)
    self.declare_parameter("pause_after_detect_sec", 0.5)
    self.declare_parameter("detect_if_within_m", 2.5)
    self.declare_parameter("stuck_window_sec", 3.5)
    self.declare_parameter("stuck_move_m", 0.15)
    self.declare_parameter("stuck_grace_sec", 3.0)
    self.declare_parameter("max_stuck_recoveries", 4)
    self.declare_parameter("stuck_backup_m", 2.5)
    self.declare_parameter("stuck_backup_wait_sec", 5.0)
    self.declare_parameter("nudge_distance_m", 1.5)
    self.declare_parameter("nudge_wait_sec", 3.0)
    self.declare_parameter("scene_name", "live_scene")
    self.declare_parameter("graph_output_dir", "/tmp/vlm_live_captures")
    self.declare_parameter("pipeline_a_export_root", "/tmp/vla3d_live")
    self.declare_parameter("export_pipeline_a", True)

    self._num_viewpoints = int(self.get_parameter("num_viewpoints").value)
    self._auto_num_viewpoints = self.get_parameter("auto_num_viewpoints").get_parameter_value().bool_value
    self._include_start = self.get_parameter("include_start_pose").get_parameter_value().bool_value
    self._bootstrap_min_points = int(self.get_parameter("bootstrap_min_points").value)
    self._bootstrap_min_wait = self.get_parameter("bootstrap_min_wait_sec").get_parameter_value().double_value
    self._bootstrap_timeout = self.get_parameter("bootstrap_timeout_sec").get_parameter_value().double_value
    self._position_tolerance = self.get_parameter("position_tolerance_m").get_parameter_value().double_value
    self._heading_tolerance = self.get_parameter("heading_tolerance_rad").get_parameter_value().double_value
    self._skip_if_within = self.get_parameter("skip_if_within_m").get_parameter_value().double_value
    self._waypoint_timeout = self.get_parameter("waypoint_timeout_sec").get_parameter_value().double_value
    self._waypoint_republish_sec = self.get_parameter("waypoint_republish_sec").get_parameter_value().double_value
    self._no_republish_within = (
      self.get_parameter("waypoint_no_republish_within_m").get_parameter_value().double_value
    )
    self._min_settle_before_reached = (
      self.get_parameter("min_settle_before_reached_sec").get_parameter_value().double_value
    )
    self._settle_sec = self.get_parameter("settle_sec").get_parameter_value().double_value
    self._require_scan_stable = self.get_parameter("require_scan_stable").get_parameter_value().bool_value
    self._scan_stable_timeout = self.get_parameter("scan_stable_timeout_sec").get_parameter_value().double_value
    self._detect_at_each_stop = self.get_parameter("detect_at_each_stop").get_parameter_value().bool_value
    self._detect_timeout = self.get_parameter("detect_timeout_sec").get_parameter_value().double_value
    self._pause_after_detect = self.get_parameter("pause_after_detect_sec").get_parameter_value().double_value
    self._detect_if_within = self.get_parameter("detect_if_within_m").get_parameter_value().double_value
    self._stuck_window = self.get_parameter("stuck_window_sec").get_parameter_value().double_value
    self._stuck_move_m = self.get_parameter("stuck_move_m").get_parameter_value().double_value
    self._stuck_grace = self.get_parameter("stuck_grace_sec").get_parameter_value().double_value
    self._max_stuck_recoveries = int(self.get_parameter("max_stuck_recoveries").value)
    self._stuck_backup_m = self.get_parameter("stuck_backup_m").get_parameter_value().double_value
    self._stuck_backup_wait_sec = (
      self.get_parameter("stuck_backup_wait_sec").get_parameter_value().double_value
    )
    self._nudge_distance = self.get_parameter("nudge_distance_m").get_parameter_value().double_value
    self._nudge_wait = self.get_parameter("nudge_wait_sec").get_parameter_value().double_value
    self._scene_name = self.get_parameter("scene_name").get_parameter_value().string_value.strip() or "live_scene"
    self._graph_output_dir = self.get_parameter("graph_output_dir").get_parameter_value().string_value
    self._pipeline_a_export_root = (
      self.get_parameter("pipeline_a_export_root").get_parameter_value().string_value
    )
    self._export_pipeline_a = self.get_parameter("export_pipeline_a").get_parameter_value().bool_value
    self._auto_start = self.get_parameter("auto_start").get_parameter_value().bool_value

    self._planner_cfg = ViewpointPlannerConfig(
      num_viewpoints=self._num_viewpoints,
      include_start_pose=self._include_start,
      min_viewpoint_spacing_m=self.get_parameter("min_viewpoint_spacing_m").get_parameter_value().double_value,
      grid_resolution_m=self.get_parameter("grid_resolution_m").get_parameter_value().double_value,
      free_clearance_m=self.get_parameter("free_clearance_m").get_parameter_value().double_value,
      wall_inset_m=self.get_parameter("wall_inset_m").get_parameter_value().double_value,
      z_min_m=self.get_parameter("z_min_m").get_parameter_value().double_value,
      z_max_m=self.get_parameter("z_max_m").get_parameter_value().double_value,
      occupied_count_min=int(self.get_parameter("occupied_count_min").value),
      max_plan_radius_m=self.get_parameter("max_plan_radius_m").get_parameter_value().double_value,
      candidate_stride_m=self.get_parameter("candidate_stride_m").get_parameter_value().double_value,
    )

    self._scan_monitor = ScanStabilityMonitor(
      window_sec=self.get_parameter("scan_stable_window_sec").get_parameter_value().double_value,
      change_threshold=self.get_parameter("scan_change_threshold").get_parameter_value().double_value,
    )

    self.vehicle_x = 0.0
    self.vehicle_y = 0.0
    self.vehicle_yaw = 0.0
    self._pose_received = False
    self._latest_scan: PointCloud2 | None = None
    self._awaiting_detection = False
    self._detection_complete_flag = False
    self._scene_graph_complete_flag = False
    self._visited: list[Viewpoint] = []
    self._failed: list[Viewpoint] = []

    self._waypoint_pub = self.create_publisher(Pose2D, WAYPOINT_TOPIC, OUTPUT_QOS)
    self._complete_pub = self.create_publisher(Bool, EXPLORATION_COMPLETE_TOPIC, COMPLETE_QOS)
    self._run_detection_pub = self.create_publisher(Bool, RUN_DETECTION_TOPIC, LIVE_QOS)
    self._marker_pub = self.create_publisher(MarkerArray, VIEWPOINT_MARKERS_TOPIC, OUTPUT_QOS)

    self.create_subscription(
      Odometry,
      STATE_ESTIMATION_TOPIC,
      self._pose_callback,
      qos_profile_sensor_data,
    )
    self.create_subscription(
      PointCloud2,
      REGISTERED_SCAN_TOPIC,
      self._scan_callback,
      qos_profile_sensor_data,
    )
    self.create_subscription(
      Bool,
      DETECTION_COMPLETE_TOPIC,
      self._detection_complete_callback,
      LIVE_QOS,
    )
    self.create_subscription(
      Bool,
      SCENE_GRAPH_COMPLETE_TOPIC,
      self._scene_graph_complete_callback,
      LIVE_QOS,
    )

    self.get_logger().info(
      f"Coverage explorer ready | num_viewpoints={self._num_viewpoints} | "
      f"spacing={self._planner_cfg.min_viewpoint_spacing_m:.2f}m | "
      f"plan_radius={self._planner_cfg.max_plan_radius_m:.1f}m | "
      f"reach={self._position_tolerance:.2f}m | detect_within={self._detect_if_within:.2f}m | "
      f"stuck_window={self._stuck_window:.1f}s | export_pipeline_a={self._export_pipeline_a}"
    )

  def _pose_callback(self, msg: Odometry) -> None:
    self.vehicle_x = msg.pose.pose.position.x
    self.vehicle_y = msg.pose.pose.position.y
    self.vehicle_yaw = yaw_from_odometry(msg)
    if not self._pose_received:
      self._pose_received = True
      self.get_logger().info(
        f"Receiving {STATE_ESTIMATION_TOPIC} at "
        f"({self.vehicle_x:.2f}, {self.vehicle_y:.2f}, yaw={math.degrees(self.vehicle_yaw):.1f}°)"
      )

  def _scan_callback(self, msg: PointCloud2) -> None:
    self._latest_scan = msg
    self._scan_monitor.update(msg)

  def _detection_complete_callback(self, msg: Bool) -> None:
    if msg.data and self._awaiting_detection:
      self._detection_complete_flag = True

  def _scene_graph_complete_callback(self, msg: Bool) -> None:
    if msg.data:
      self._scene_graph_complete_flag = True

  def run_exploration(self) -> None:
    if not self._wait_for_pose(timeout_sec=30.0):
      self.get_logger().error(f"No pose on {STATE_ESTIMATION_TOPIC}; aborting exploration.")
      return

    self._wait_for_bootstrap()
    self._maybe_autoscale_viewpoints()
    self._visited = [Viewpoint(self.vehicle_x, self.vehicle_y)]
    self._failed = []

    preview = self._plan_viewpoints()
    self._publish_viewpoint_markers(preview)
    self.get_logger().info(
      f"Coverage tour preview ({len(preview)} snaps, budget={self._num_viewpoints}): "
      + ", ".join(f"({v.x:.2f},{v.y:.2f})" for v in preview)
      + ". Next stop is always the farthest uncovered free cell "
      "(not a nearest-neighbor replan). Ensure sim is in RViz Waypoint mode."
    )

    snaps_done = 0
    visit_index = 0
    if self._include_start:
      self.get_logger().info(
        f"Detecting at start {snaps_done + 1}/{self._num_viewpoints}: "
        f"robot=({self.vehicle_x:.2f},{self.vehicle_y:.2f})"
      )
      self._settle_at_stop(0)
      if self._detect_at_each_stop:
        self._trigger_detection(0, self._visited[0])
      snaps_done += 1
      visit_index += 1

    while snaps_done < self._num_viewpoints and rclpy.ok():
      view = self._plan_next_stop()
      if view is None:
        self.get_logger().info("No uncovered standpoint left — ending tour")
        break

      visit_index += 1
      tour_markers = self._visited + [view]
      self._publish_viewpoint_markers(tour_markers)
      dist_start = math.hypot(self.vehicle_x - view.x, self.vehicle_y - view.y)
      if dist_start <= self._skip_if_within:
        self.get_logger().info(
          f"Next stop ({view.x:.2f},{view.y:.2f}) is {dist_start:.2f}m away — "
          "already covered, stopping tour"
        )
        break

      reached = self._visit_viewpoint(visit_index - 1, view, [view])
      dist = math.hypot(self.vehicle_x - view.x, self.vehicle_y - view.y)
      progress = dist_start - dist
      detect_radius = max(self._detect_if_within, 0.35 * max(dist_start, 0.1))
      close_enough = reached or dist <= detect_radius or progress >= 1.0
      here = Viewpoint(self.vehicle_x, self.vehicle_y)
      if not close_enough:
        self.get_logger().warn(
          f"Skip detect at view {visit_index}: robot=({self.vehicle_x:.2f},{self.vehicle_y:.2f}) "
          f"dist={dist:.2f}m — blacklisting that goal, picking a new far stop"
        )
        self._failed.append(view)
        if progress >= 1.0:
          self._visited.append(here)
        continue

      self.get_logger().info(
        f"Detecting at stop {snaps_done + 1}/{self._num_viewpoints}: "
        f"robot=({self.vehicle_x:.2f},{self.vehicle_y:.2f}) "
        f"dist_to_goal={dist:.2f}m"
      )
      self._settle_at_stop(visit_index - 1)
      if self._detect_at_each_stop:
        self._trigger_detection(visit_index - 1, view)
      snaps_done += 1
      self._visited.append(here)

    self._export_pipeline_a_scene()
    complete = Bool()
    complete.data = True
    self._complete_pub.publish(complete)
    write_token(HS_EXPLORATION_COMPLETE, 1)
    self.get_logger().info(
      f"Exploration complete — published {EXPLORATION_COMPLETE_TOPIC}=true"
    )

    if self.get_parameter("shutdown_on_complete").get_parameter_value().bool_value:
      self.get_logger().info("Explorer shutting down.")
      rclpy.shutdown()

  def _wait_for_bootstrap(self) -> None:
    started = time.monotonic()
    deadline = started + self._bootstrap_timeout
    last_log = started
    while time.monotonic() < deadline and rclpy.ok():
      self._idle(0.1)
      waited = time.monotonic() - started
      n_pts = point_cloud_size(self._latest_scan) if self._latest_scan is not None else 0
      if waited >= self._bootstrap_min_wait and n_pts >= self._bootstrap_min_points:
        self.get_logger().info(
          f"Bootstrap scan ready after {waited:.1f}s ({n_pts} points)"
        )
        return
      now = time.monotonic()
      if now - last_log >= 4.0:
        self.get_logger().info(
          f"Waiting for /registered_scan bootstrap "
          f"({n_pts}/{self._bootstrap_min_points} points, {waited:.1f}s)"
        )
        last_log = now

    n_pts = point_cloud_size(self._latest_scan) if self._latest_scan is not None else 0
    self.get_logger().warn(
      f"Bootstrap timed out with {n_pts} points — planning from current cloud"
    )

  def _maybe_autoscale_viewpoints(self) -> None:
    if not self._auto_num_viewpoints:
      return
    points = self._scan_xyz()
    suggested = suggested_viewpoint_count(
      points,
      spacing_m=self._planner_cfg.min_viewpoint_spacing_m,
      min_n=5,
      max_n=max(8, self._num_viewpoints),
    )
    if suggested > self._num_viewpoints:
      self.get_logger().info(
        f"Long scanned room — raising snap budget {self._num_viewpoints} → {suggested}"
      )
      self._num_viewpoints = suggested

  def _plan_viewpoints(self) -> list[Viewpoint]:
    robot = (self.vehicle_x, self.vehicle_y)
    points = self._scan_xyz()
    views = plan_coverage_viewpoints(
      points, robot, self._planner_cfg, visited=self._visited, exclude=self._failed
    )
    if not views:
      views = [Viewpoint(self.vehicle_x, self.vehicle_y)]
    return order_nearest_neighbor(views, robot)

  def _plan_next_stop(self) -> Viewpoint | None:
    """Farthest free cell from every snap already taken (map may have grown)."""
    robot = (self.vehicle_x, self.vehicle_y)
    view = plan_next_coverage_viewpoint(
      self._scan_xyz(),
      robot,
      self._visited,
      self._planner_cfg,
      exclude=self._failed,
    )
    if view is None:
      self.get_logger().warn("No further uncovered viewpoint in the current cloud")
      return None
    nearest = min(
      math.hypot(view.x - v.x, view.y - v.y) for v in self._visited
    ) if self._visited else 0.0
    self.get_logger().info(
      f"Next uncovered stop ({view.x:.2f},{view.y:.2f}) "
      f"{nearest:.1f}m from nearest previous snap"
    )
    return view

  def _scan_xyz(self):
    if self._latest_scan is None:
      return None
    try:
      return pointcloud2_to_xyz(self._latest_scan)
    except Exception as exc:
      self.get_logger().warn(f"Failed to parse /registered_scan: {exc}")
      return None

  def _visit_viewpoint(
    self,
    index: int,
    view: Viewpoint,
    tour: list[Viewpoint],
  ) -> bool:
    dist = math.hypot(self.vehicle_x - view.x, self.vehicle_y - view.y)
    if dist <= self._skip_if_within:
      self.get_logger().info(
        f"View {index + 1}/{len(tour)}: already at ({view.x:.2f}, {view.y:.2f}) "
        f"dist={dist:.2f}m — no drive"
      )
      return True

    heading = self._travel_heading(view, tour, index)
    self.get_logger().info(
      f"View {index + 1}/{len(tour)}: drive to ({view.x:.2f}, {view.y:.2f}) "
      f"from ({self.vehicle_x:.2f}, {self.vehicle_y:.2f}) heading={math.degrees(heading):.0f}°"
    )
    self._publish_active_goal_marker(view)
    self._publish_waypoint(view.x, view.y, heading, log_publish=True)
    if self._wait_for_waypoint(view.x, view.y, heading):
      return True
    self.get_logger().warn(
      f"Stopped short of view {index + 1} ({view.x:.2f}, {view.y:.2f}) "
      f"at robot=({self.vehicle_x:.2f}, {self.vehicle_y:.2f})"
    )
    return False

  def _travel_heading(self, view: Viewpoint, tour: list[Viewpoint], index: int) -> float:
    dx = view.x - self.vehicle_x
    dy = view.y - self.vehicle_y
    if math.hypot(dx, dy) < 0.05 and index + 1 < len(tour):
      nxt = tour[index + 1]
      dx = nxt.x - view.x
      dy = nxt.y - view.y
    if math.hypot(dx, dy) < 0.05:
      return self.vehicle_yaw
    return math.atan2(dy, dx)

  def _settle_at_stop(self, index: int) -> None:
    if self._require_scan_stable:
      self._scan_monitor.reset()
      if self._wait_for_scan_stable():
        self.get_logger().info(f"Scan stable at view {index + 1}")
      else:
        self.get_logger().warn(f"Scan did not stabilise at view {index + 1}; detecting anyway")
      return
    if self._settle_sec > 0:
      self.get_logger().info(f"Settling {self._settle_sec:.1f}s at view {index + 1}...")
      self._idle(self._settle_sec)

  def _trigger_detection(self, index: int, view: Viewpoint) -> None:
    self._detection_complete_flag = False
    self._scene_graph_complete_flag = False
    self._awaiting_detection = True
    token = bump(HS_RUN_DETECTION)
    msg = Bool()
    msg.data = True
    self._run_detection_pub.publish(msg)
    last_pub = time.monotonic()
    self.get_logger().info(
      f"Triggered {RUN_DETECTION_TOPIC} at view {index + 1} "
      f"({self.vehicle_x:.2f}, {self.vehicle_y:.2f}) target=({view.x:.2f}, {view.y:.2f}) "
      f"| token={token} subscribers={self._run_detection_pub.get_subscription_count()}"
    )

    deadline = time.monotonic() + self._detect_timeout
    while time.monotonic() < deadline and rclpy.ok():
      self._idle(0.1)
      if (
        self._detection_complete_flag
        or read_token(HS_DETECTION_COMPLETE) >= token
      ):
        self._detection_complete_flag = True
        self.get_logger().info(f"Detection finished at view {index + 1}")
        self._awaiting_detection = False
        graph_deadline = time.monotonic() + 8.0
        while (
          time.monotonic() < graph_deadline
          and rclpy.ok()
          and not self._scene_graph_complete_flag
          and read_token(HS_SCENE_GRAPH_COMPLETE) < token
        ):
          self._idle(0.1)
        if self._pause_after_detect > 0:
          self._idle(self._pause_after_detect)
        return
      now = time.monotonic()
      if now - last_pub >= 2.0:
        self._run_detection_pub.publish(msg)
        last_pub = now

    self._awaiting_detection = False
    self.get_logger().warn(
      f"Timed out waiting for {DETECTION_COMPLETE_TOPIC} at view {index + 1}"
    )

  def _publish_waypoint(
    self,
    x: float,
    y: float,
    heading: float,
    *,
    log_publish: bool = False,
  ) -> None:
    waypoint = Pose2D()
    waypoint.x = x
    waypoint.y = y
    waypoint.theta = heading
    self._waypoint_pub.publish(waypoint)
    if log_publish:
      self.get_logger().info(
        f"Published {WAYPOINT_TOPIC} ({x:.2f}, {y:.2f}, θ={math.degrees(heading):.0f}°)"
      )

  def _publish_viewpoint_markers(self, viewpoints: list[Viewpoint]) -> None:
    array = MarkerArray()
    now = self.get_clock().now().to_msg()
    delete = Marker()
    delete.header.frame_id = "map"
    delete.header.stamp = now
    delete.ns = "explorer_viewpoints"
    delete.id = 0
    delete.action = Marker.DELETEALL
    array.markers.append(delete)

    for index, view in enumerate(viewpoints):
      marker = Marker()
      marker.header.frame_id = "map"
      marker.header.stamp = now
      marker.ns = "explorer_viewpoints"
      marker.id = index + 1
      marker.action = Marker.ADD
      marker.type = Marker.SPHERE
      marker.pose.position.x = view.x
      marker.pose.position.y = view.y
      marker.pose.position.z = 0.4
      marker.pose.orientation.w = 1.0
      marker.scale.x = 0.35
      marker.scale.y = 0.35
      marker.scale.z = 0.35
      if index == 0:
        marker.color = ColorRGBA(r=0.1, g=0.85, b=0.25, a=0.95)
      else:
        marker.color = ColorRGBA(r=0.1, g=0.65, b=1.0, a=0.95)
      array.markers.append(marker)

    path = Marker()
    path.header.frame_id = "map"
    path.header.stamp = now
    path.ns = "explorer_viewpoints"
    path.id = 1000
    path.action = Marker.ADD
    path.type = Marker.LINE_STRIP
    path.pose.orientation.w = 1.0
    path.scale.x = 0.06
    path.color = ColorRGBA(r=0.15, g=0.75, b=1.0, a=0.7)
    for view in viewpoints:
      p = Point()
      p.x = view.x
      p.y = view.y
      p.z = 0.15
      path.points.append(p)
    array.markers.append(path)
    self._marker_pub.publish(array)

  def _publish_active_goal_marker(self, view: Viewpoint) -> None:
    """Stable cyan goal — does not jump when autonomy snaps /way_point."""
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = self.get_clock().now().to_msg()
    marker.ns = "explorer_active_goal"
    marker.id = 1
    marker.action = Marker.ADD
    marker.type = Marker.SPHERE
    marker.pose.position.x = view.x
    marker.pose.position.y = view.y
    marker.pose.position.z = 0.55
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.55
    marker.scale.y = 0.55
    marker.scale.z = 0.55
    marker.color = ColorRGBA(r=0.0, g=0.95, b=0.95, a=0.95)
    array = MarkerArray()
    array.markers.append(marker)
    self._marker_pub.publish(array)

  def _wait_for_pose(self, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and rclpy.ok():
      if self._pose_received:
        return True
      self._idle(0.1)
    return self._pose_received

  def _wait_for_waypoint(self, target_x: float, target_y: float, target_heading: float) -> bool:
    """Drive toward ``target``. If wedged, back up, then republish the target."""
    deadline = time.monotonic() + self._waypoint_timeout
    wait_started = time.monotonic()
    last_publish = wait_started
    last_log = wait_started
    last_motion_x = self.vehicle_x
    last_motion_y = self.vehicle_y
    last_motion_time = wait_started
    recoveries = 0
    final_x, final_y = target_x, target_y
    active_heading = target_heading

    while time.monotonic() < deadline and rclpy.ok():
      self._idle(0.1)
      now = time.monotonic()
      dist_final = math.hypot(self.vehicle_x - final_x, self.vehicle_y - final_y)
      waited_sec = now - wait_started

      moved = math.hypot(self.vehicle_x - last_motion_x, self.vehicle_y - last_motion_y)
      if moved >= self._stuck_move_m:
        last_motion_x = self.vehicle_x
        last_motion_y = self.vehicle_y
        last_motion_time = now

      if waited_sec >= self._min_settle_before_reached and dist_final <= self._position_tolerance:
        self.get_logger().info(
          f"Reached view waypoint ({final_x:.2f}, {final_y:.2f}) "
          f"(pos_err={dist_final:.2f}m)"
        )
        return True

      stuck = (
        waited_sec >= self._stuck_grace
        and (now - last_motion_time) >= self._stuck_window
      )
      if stuck:
        if dist_final <= self._detect_if_within:
          self.get_logger().warn(
            f"Stuck at ({self.vehicle_x:.2f}, {self.vehicle_y:.2f}) but within "
            f"{dist_final:.2f}m of goal — treating as close enough"
          )
          return True
        if recoveries < self._max_stuck_recoveries:
          recoveries += 1
          if self._unstick_with_backup(
            final_x, final_y, recovery_index=recoveries, deadline=deadline
          ):
            return True
          active_heading = math.atan2(final_y - self.vehicle_y, final_x - self.vehicle_x)
          last_publish = time.monotonic()
          last_motion_x = self.vehicle_x
          last_motion_y = self.vehicle_y
          last_motion_time = last_publish
          continue
        self.get_logger().warn(
          f"Still stuck after {recoveries} recoveries; "
          f"robot=({self.vehicle_x:.2f}, {self.vehicle_y:.2f}) "
          f"goal=({final_x:.2f}, {final_y:.2f})"
        )
        return False

      if (
        dist_final > self._no_republish_within
        and now - last_publish >= self._waypoint_republish_sec
      ):
        self._publish_waypoint(final_x, final_y, active_heading)
        last_publish = now

      if now - last_log >= 5.0:
        self.get_logger().info(
          f"En route: robot=({self.vehicle_x:.2f}, {self.vehicle_y:.2f}), "
          f"final=({final_x:.2f}, {final_y:.2f}), "
          f"pos_err={dist_final:.2f}m, still={now - last_motion_time:.1f}s"
        )
        last_log = now
    return False

  def _unstick_with_backup(
    self,
    final_x: float,
    final_y: float,
    *,
    recovery_index: int,
    deadline: float,
  ) -> bool:
    """Publish a waypoint behind the robot, wait for motion, then retry the target."""
    backup_m = max(self._stuck_backup_m, self._position_tolerance + 0.5)
    bx, by, btheta = backup_waypoint(
      self.vehicle_x,
      self.vehicle_y,
      self.vehicle_yaw,
      final_x,
      final_y,
      backup_m=backup_m,
      recovery_index=recovery_index,
    )
    self.get_logger().warn(
      f"Stuck at ({self.vehicle_x:.2f}, {self.vehicle_y:.2f}) "
      f"goal=({final_x:.2f}, {final_y:.2f}) — backing up to "
      f"({bx:.2f}, {by:.2f}) recovery {recovery_index}/{self._max_stuck_recoveries}"
    )
    self._publish_waypoint(bx, by, btheta, log_publish=True)
    backup_deadline = min(deadline, time.monotonic() + self._stuck_backup_wait_sec)
    start_x, start_y = self.vehicle_x, self.vehicle_y
    while time.monotonic() < backup_deadline and rclpy.ok():
      self._idle(0.1)
      if math.hypot(self.vehicle_x - final_x, self.vehicle_y - final_y) <= self._position_tolerance:
        return True
      if math.hypot(self.vehicle_x - start_x, self.vehicle_y - start_y) >= self._stuck_move_m:
        break

    heading = math.atan2(final_y - self.vehicle_y, final_x - self.vehicle_x)
    self.get_logger().info(
      f"Retrying original target ({final_x:.2f}, {final_y:.2f}) after backup"
    )
    self._publish_waypoint(final_x, final_y, heading, log_publish=True)
    return False

  def _export_pipeline_a_scene(self) -> None:
    if not self._export_pipeline_a:
      return
    self._idle(1.0)
    if not self._scene_graph_complete_flag:
      self.get_logger().info("Waiting up to 15s for scene graph to finish writing...")
      deadline = time.monotonic() + 15.0
      while (
        time.monotonic() < deadline
        and rclpy.ok()
        and not self._scene_graph_complete_flag
        and read_token(HS_SCENE_GRAPH_COMPLETE) <= 0
      ):
        self._idle(0.2)

    latest = Path(self._graph_output_dir).expanduser() / self._scene_name / "latest_scene_graph.json"
    if not latest.is_file():
      self.get_logger().error(
        f"No live scene graph at {latest} — skip Pipeline A export. "
        "Did any detection succeed?"
      )
      return

    out_dir = Path(self._pipeline_a_export_root).expanduser() / self._scene_name
    try:
      written = export_scene_folder(latest, out_dir, self._scene_name)
    except Exception as exc:
      self.get_logger().error(f"Pipeline A export failed: {exc}")
      return

    csv_path = written.get("csv")
    graph_path = written.get("scene_graph")
    self.get_logger().info(f"Exported Pipeline A CSV → {csv_path}")
    self.get_logger().info(f"Exported Pipeline A scene graph → {graph_path}")
    self.get_logger().info(
      "Launch Pipeline A with:\n"
      f"  ros2 launch vlm_pipeline vlm_pipeline.launch.py "
      f"scene_name:={self._scene_name} "
      f"vla3d_data_root:={self._pipeline_a_export_root}"
    )

  def _wait_for_scan_stable(self) -> bool:
    deadline = time.monotonic() + self._scan_stable_timeout
    while time.monotonic() < deadline and rclpy.ok():
      self._idle(0.1)
      if self._latest_scan is not None and self._scan_monitor.is_stable():
        return True
    return self._scan_monitor.is_stable()

  def _idle(self, duration_sec: float) -> None:
    """Wait while the executor thread delivers pose / scan / detect-complete."""
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline and rclpy.ok():
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        break
      time.sleep(min(0.05, remaining))


def main(args=None) -> None:
  rclpy.init(args=args)
  node = ExplorerNode()
  executor = MultiThreadedExecutor(num_threads=4)
  executor.add_node(node)
  spin_thread = threading.Thread(target=executor.spin, daemon=True)
  spin_thread.start()
  try:
    node.run_exploration()
  except KeyboardInterrupt:
    pass
  finally:
    executor.shutdown()
    node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == "__main__":
  main()
