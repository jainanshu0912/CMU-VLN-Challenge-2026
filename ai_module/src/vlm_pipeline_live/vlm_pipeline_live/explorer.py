"""Rotate-in-place exploration for Pipeline B scene mapping."""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool

from vlm_pipeline_live.scan_stability import ScanStabilityMonitor

STATE_ESTIMATION_TOPIC = "/state_estimation"
REGISTERED_SCAN_TOPIC = "/registered_scan"
WAYPOINT_TOPIC = "/way_point_with_heading"
EXPLORATION_COMPLETE_TOPIC = "/vlm_live/exploration_complete"

DEFAULT_HEADINGS_DEG = (0.0, 90.0, 180.0, 270.0)
OUTPUT_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


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
  """Publish four rotate-in-place waypoints and wait for scan stabilisation."""

  def __init__(self) -> None:
    super().__init__("vlm_live_explorer")

    self.declare_parameter("headings_deg", list(DEFAULT_HEADINGS_DEG))
    self.declare_parameter("rotation_standoff_m", 0.8)
    self.declare_parameter("position_tolerance_m", 0.2)
    self.declare_parameter("heading_tolerance_rad", 0.35)
    self.declare_parameter("min_settle_before_reached_sec", 1.0)
    self.declare_parameter("heading_wait_timeout_sec", 120.0)
    self.declare_parameter("waypoint_republish_sec", 2.0)
    self.declare_parameter("scan_stable_window_sec", 5.0)
    self.declare_parameter("scan_change_threshold", 0.05)
    self.declare_parameter("scan_stable_timeout_sec", 30.0)
    self.declare_parameter("require_scan_stable", False)
    self.declare_parameter("scan_settle_sec", 6.0)
    self.declare_parameter("pause_after_view_sec", 1.0)
    self.declare_parameter("auto_start", True)
    self.declare_parameter("shutdown_on_complete", True)

    headings_deg = self.get_parameter("headings_deg").value
    self._headings_rad = [math.radians(float(h)) for h in headings_deg]
    self._rotation_standoff = (
      self.get_parameter("rotation_standoff_m").get_parameter_value().double_value
    )
    self._position_tolerance = (
      self.get_parameter("position_tolerance_m").get_parameter_value().double_value
    )
    self._heading_tolerance = (
      self.get_parameter("heading_tolerance_rad").get_parameter_value().double_value
    )
    self._heading_wait_timeout = (
      self.get_parameter("heading_wait_timeout_sec").get_parameter_value().double_value
    )
    self._waypoint_republish_sec = (
      self.get_parameter("waypoint_republish_sec").get_parameter_value().double_value
    )
    self._min_settle_before_reached = (
      self.get_parameter("min_settle_before_reached_sec").get_parameter_value().double_value
    )
    self._scan_stable_timeout = (
      self.get_parameter("scan_stable_timeout_sec").get_parameter_value().double_value
    )
    self._pause_after_view = (
      self.get_parameter("pause_after_view_sec").get_parameter_value().double_value
    )
    self._require_scan_stable = (
      self.get_parameter("require_scan_stable").get_parameter_value().bool_value
    )
    self._scan_settle_sec = (
      self.get_parameter("scan_settle_sec").get_parameter_value().double_value
    )
    self._auto_start = self.get_parameter("auto_start").get_parameter_value().bool_value

    self._scan_monitor = ScanStabilityMonitor(
      window_sec=self.get_parameter("scan_stable_window_sec").get_parameter_value().double_value,
      change_threshold=self.get_parameter("scan_change_threshold").get_parameter_value().double_value,
    )

    self.vehicle_x = 0.0
    self.vehicle_y = 0.0
    self.vehicle_yaw = 0.0
    self._pose_received = False
    self._latest_scan: PointCloud2 | None = None

    self._waypoint_pub = self.create_publisher(Pose2D, WAYPOINT_TOPIC, OUTPUT_QOS)
    self._complete_pub = self.create_publisher(Bool, EXPLORATION_COMPLETE_TOPIC, OUTPUT_QOS)

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

    self.get_logger().info(
      f"Explorer ready | headings_deg={headings_deg} | "
      f"rotation_standoff_m={self._rotation_standoff:.2f} | "
      f"scan_settle_sec={self._scan_settle_sec:.1f} | "
      f"require_scan_stable={self._require_scan_stable}"
    )

    if self._auto_start:
      self._exploration_timer = self.create_timer(1.0, self._start_exploration_once)

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

  def _start_exploration_once(self) -> None:
    self._exploration_timer.cancel()
    self.run_exploration()

  def run_exploration(self) -> None:
    if not self._wait_for_pose(timeout_sec=30.0):
      self.get_logger().error(f"No pose on {STATE_ESTIMATION_TOPIC}; aborting exploration.")
      return

    anchor_x = self.vehicle_x
    anchor_y = self.vehicle_y
    total_views = len(self._headings_rad)

    self.get_logger().info(
      f"Starting exploration at ({anchor_x:.2f}, {anchor_y:.2f}) for {total_views} views. "
      f"Ensure sim is in RViz Waypoint mode (Resume Navigation to Goal)."
    )

    for index, heading in enumerate(self._headings_rad):
      base_x = self.vehicle_x
      base_y = self.vehicle_y
      wp_x = base_x + self._rotation_standoff * math.cos(heading)
      wp_y = base_y + self._rotation_standoff * math.sin(heading)
      self.get_logger().info(
        f"View {index + 1}/{total_views}: heading {math.degrees(heading):.0f}° "
        f"from ({base_x:.2f}, {base_y:.2f}) → waypoint ({wp_x:.2f}, {wp_y:.2f})"
      )
      self._publish_rotate_waypoint(wp_x, wp_y, heading, log_publish=True)

      if not self._wait_for_waypoint(wp_x, wp_y, heading):
        self.get_logger().warn(
          f"Timed out waiting for view {index + 1} "
          f"(heading {math.degrees(heading):.0f}°). "
          "Check RViz 'Resume Navigation to Goal', /way_point output, and "
          "waypointConverter logs for 'Waypoint not in traversable area'."
        )

      self._scan_monitor.reset()
      if self._require_scan_stable:
        if not self._wait_for_scan_stable():
          self.get_logger().warn(
            f"Scan did not stabilise after view {index + 1}; continuing anyway"
          )
        else:
          self.get_logger().info(
            f"Scan stable after view {index + 1} "
            f"(relative_change={self._scan_monitor.relative_change():.3f})"
          )
      elif self._scan_settle_sec > 0:
        self.get_logger().info(
          f"Settling scan for {self._scan_settle_sec:.1f}s after view {index + 1}..."
        )
        self._sleep_with_spin(self._scan_settle_sec)

      if self._pause_after_view > 0 and index < total_views - 1:
        self._sleep_with_spin(self._pause_after_view)

    complete = Bool()
    complete.data = True
    self._complete_pub.publish(complete)
    self.get_logger().info(
      f"Exploration complete — published {EXPLORATION_COMPLETE_TOPIC}=true"
    )

    if self.get_parameter("shutdown_on_complete").get_parameter_value().bool_value:
      self.get_logger().info("Explorer shutting down.")
      rclpy.shutdown()

  def _publish_rotate_waypoint(
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

  def _wait_for_pose(self, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and rclpy.ok():
      if self._pose_received:
        return True
      rclpy.spin_once(self, timeout_sec=0.1)
    return self._pose_received

  def _wait_for_waypoint(
    self,
    target_x: float,
    target_y: float,
    target_heading: float,
  ) -> bool:
    deadline = time.monotonic() + self._heading_wait_timeout
    wait_started = time.monotonic()
    last_publish = wait_started
    last_log = wait_started

    while time.monotonic() < deadline and rclpy.ok():
      rclpy.spin_once(self, timeout_sec=0.1)

      now = time.monotonic()
      if now - last_publish >= self._waypoint_republish_sec:
        self._publish_rotate_waypoint(target_x, target_y, target_heading)
        last_publish = now

      position_error = math.hypot(self.vehicle_x - target_x, self.vehicle_y - target_y)
      heading_error = angular_error(self.vehicle_yaw, target_heading)
      waited_sec = now - wait_started
      if (
        waited_sec >= self._min_settle_before_reached
        and position_error <= self._position_tolerance
        and heading_error <= self._heading_tolerance
      ):
        self.get_logger().info(
          f"Reached view waypoint ({target_x:.2f}, {target_y:.2f}) "
          f"(pos_err={position_error:.2f}m, yaw_err={math.degrees(heading_error):.1f}°)"
        )
        return True

      if now - last_log >= 5.0:
        self.get_logger().info(
          f"En route: robot=({self.vehicle_x:.2f}, {self.vehicle_y:.2f}, "
          f"yaw={math.degrees(self.vehicle_yaw):.0f}°), "
          f"target=({target_x:.2f}, {target_y:.2f}, "
          f"θ={math.degrees(target_heading):.0f}°), "
          f"pos_err={position_error:.2f}m, yaw_err={math.degrees(heading_error):.1f}°"
        )
        last_log = now

    return False

  def _wait_for_scan_stable(self) -> bool:
    deadline = time.monotonic() + self._scan_stable_timeout
    while time.monotonic() < deadline and rclpy.ok():
      rclpy.spin_once(self, timeout_sec=0.1)
      if self._latest_scan is not None and self._scan_monitor.is_stable():
        return True
    return self._scan_monitor.is_stable()

  def _sleep_with_spin(self, duration_sec: float) -> None:
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline and rclpy.ok():
      rclpy.spin_once(self, timeout_sec=0.1)


def main(args=None) -> None:
  rclpy.init(args=args)
  node = ExplorerNode()
  try:
    if not node.get_parameter("auto_start").get_parameter_value().bool_value:
      node.run_exploration()
    else:
      rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == "__main__":
  main()
