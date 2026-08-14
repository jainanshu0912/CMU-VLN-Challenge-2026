"""ROS 2 entry point for Pipeline C (SORT3D-style reasoning).

Independent of Pipelines A and B. Publishes the same challenge output topics.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from vlm_pipeline_sort3d.object_captioner import ObjectCaptioner
from vlm_pipeline_sort3d.object_filter import ObjectFilter
from vlm_pipeline_sort3d.question_classifier import QuestionType, classify_question
from vlm_pipeline_sort3d.scene_inventory import load_vla3d_inventory
from vlm_pipeline_sort3d.spatial_toolbox import SpatialToolbox, ToolboxConfig
from vlm_pipeline_sort3d.toolbox_reasoner import ToolboxReasoner


class Sort3dNode(Node):
  def __init__(self) -> None:
    super().__init__("sort3d_node")

    self.declare_parameter("scene_name", "chinese_room")
    self.declare_parameter("vla3d_root", "")
    self.declare_parameter("near_threshold_m", 1.5)
    self.declare_parameter("on_contact_threshold_m", 0.05)
    self.declare_parameter("between_threshold_m", 0.5)
    self.declare_parameter("standoff_m", 0.8)
    self.declare_parameter("use_vlm_captions", False)

    self._robot_pose: Optional[Tuple[float, float, float]] = None
    self._inventory = None
    self._load_inventory()

    self._filter = ObjectFilter(llm=None)  # wire LLM backend later
    self._captioner = ObjectCaptioner()
    self._reasoner = ToolboxReasoner(llm=None)

    self.create_subscription(String, "/challenge_question", self._on_question, 10)
    self.create_subscription(Odometry, "/state_estimation", self._on_odom, 10)

    self._pub_marker = self.create_publisher(Marker, "/selected_object_marker", 10)
    self._pub_count = self.create_publisher(Int32, "/numerical_response", 10)
    self._pub_wp = self.create_publisher(Pose2D, "/way_point_with_heading", 10)

    self.get_logger().info(
      f"Pipeline C ready | scene={self.get_parameter('scene_name').value} | "
      f"objects={len(self._inventory.objects) if self._inventory else 0}"
    )

  def _load_inventory(self) -> None:
    scene = str(self.get_parameter("scene_name").value)
    root = str(self.get_parameter("vla3d_root").value).strip() or None
    try:
      self._inventory = load_vla3d_inventory(scene, vla3d_root=root)
    except FileNotFoundError as exc:
      self.get_logger().error(str(exc))
      self._inventory = None

  def _on_odom(self, msg: Odometry) -> None:
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    q = msg.pose.pose.orientation
    # yaw from quaternion (z-up)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    self._robot_pose = (x, y, yaw)

  def _on_question(self, msg: String) -> None:
    question = msg.data.strip()
    if not question:
      return
    if self._inventory is None:
      self.get_logger().error("No scene inventory loaded — cannot answer")
      return

    qtype = classify_question(question)
    self.get_logger().info(f"Q[{qtype.value}]: {question}")

    if qtype == QuestionType.NAVIGATE:
      self.get_logger().warn("NAVIGATE questions are stubbed in Pipeline C")
      return

    filt = self._filter.filter(question, self._inventory)
    inventory = self._captioner.caption_objects(filt.filtered)
    self.get_logger().info(
      f"Filter nouns={filt.nouns} kept={len(inventory.objects)}/{len(self._inventory.objects)}"
    )

    cfg = ToolboxConfig(
      near_threshold_m=float(self.get_parameter("near_threshold_m").value),
      on_contact_threshold_m=float(self.get_parameter("on_contact_threshold_m").value),
      between_threshold_m=float(self.get_parameter("between_threshold_m").value),
      standoff_m=float(self.get_parameter("standoff_m").value),
    )
    toolbox = SpatialToolbox(inventory, config=cfg, robot_pose=self._robot_pose)

    if self._reasoner.llm is None:
      self.get_logger().error(
        "ToolboxReasoner has no LLM backend yet. "
        "Wire an LLM in main_node (or use offline tests with a mock)."
      )
      return

    result = self._reasoner.reason(
      question,
      inventory,
      toolbox,
      question_type=qtype.value,
    )
    if not result.success:
      self.get_logger().error(f"Reasoning failed: {result.error}")
      for line in result.trace[-4:]:
        self.get_logger().info(line)
      return

    if result.answer_kind == "count" and result.count is not None:
      out = Int32()
      out.data = int(result.count)
      self._pub_count.publish(out)
      self.get_logger().info(f"COUNT → {result.count}")
      return

    if result.answer_kind == "find" and result.object_id is not None:
      obj = inventory.by_id().get(result.object_id) or self._inventory.by_id().get(result.object_id)
      if obj is None:
        self.get_logger().error(f"Unknown object id {result.object_id}")
        return
      marker = Marker()
      marker.header.frame_id = "map"
      marker.header.stamp = self.get_clock().now().to_msg()
      marker.ns = "sort3d"
      marker.id = 0
      marker.type = Marker.CUBE
      marker.action = Marker.ADD
      marker.pose.position.x = obj.cx
      marker.pose.position.y = obj.cy
      marker.pose.position.z = obj.cz
      marker.pose.orientation.w = 1.0
      marker.scale.x = max(obj.x_length, 0.05)
      marker.scale.y = max(obj.y_length, 0.05)
      marker.scale.z = max(obj.z_length, 0.05)
      marker.color.r = 0.1
      marker.color.g = 0.8
      marker.color.b = 0.2
      marker.color.a = 0.7
      self._pub_marker.publish(marker)

      wp = result.waypoint or toolbox.go_near(result.object_id)
      if wp is not None:
        pose = Pose2D()
        pose.x = wp.x
        pose.y = wp.y
        pose.theta = wp.yaw
        self._pub_wp.publish(pose)
      self.get_logger().info(f"FIND → id={result.object_id}")


def main(args=None) -> None:
  rclpy.init(args=args)
  node = Sort3dNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()
