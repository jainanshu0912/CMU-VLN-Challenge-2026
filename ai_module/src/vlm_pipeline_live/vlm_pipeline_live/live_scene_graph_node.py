"""ROS node: build live scene graph after detection completes."""

from __future__ import annotations

import traceback

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from vlm_pipeline_live.live_detector import detections_from_json
from vlm_pipeline_live.live_scene_graph import (
  LiveSceneGraphBuilder,
  save_scene_graph_json,
  scene_graph_to_json_string,
)

DETECTIONS_JSON_TOPIC = "/vlm_live/detections_json"
DETECTION_COMPLETE_TOPIC = "/vlm_live/detection_complete"
SCENE_GRAPH_JSON_TOPIC = "/vlm_live/scene_graph_json"
SCENE_GRAPH_COMPLETE_TOPIC = "/vlm_live/scene_graph_complete"

OUTPUT_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


class LiveSceneGraphNode(Node):
  """Subscribe to fused detections and publish a VLA-3D-format scene graph."""

  def __init__(self) -> None:
    super().__init__("vlm_live_scene_graph")

    self.declare_parameter("scene_name", "live_scene")
    self.declare_parameter("region_name", "live_map")
    self.declare_parameter("near_distance_m", 1.5)
    self.declare_parameter("auto_run_on_detection_complete", True)
    self.declare_parameter("save_graph", True)
    self.declare_parameter("graph_output_path", "/tmp/vlm_live_scene_graph.json")
    self.declare_parameter("shutdown_on_complete", False)

    self._latest_detections_json: str | None = None
    self._graph_done = False

    self._builder = LiveSceneGraphBuilder(
      scene_name=self.get_parameter("scene_name").get_parameter_value().string_value,
      region_name=self.get_parameter("region_name").get_parameter_value().string_value,
      near_distance_m=self.get_parameter("near_distance_m").get_parameter_value().double_value,
    )

    self._graph_pub = self.create_publisher(String, SCENE_GRAPH_JSON_TOPIC, OUTPUT_QOS)
    self._complete_pub = self.create_publisher(Bool, SCENE_GRAPH_COMPLETE_TOPIC, OUTPUT_QOS)

    self.create_subscription(String, DETECTIONS_JSON_TOPIC, self._detections_callback, OUTPUT_QOS)
    if self.get_parameter("auto_run_on_detection_complete").get_parameter_value().bool_value:
      self.create_subscription(
        Bool,
        DETECTION_COMPLETE_TOPIC,
        self._detection_complete_callback,
        OUTPUT_QOS,
      )

    self.get_logger().info(
      f"Live scene graph ready | waits for {DETECTION_COMPLETE_TOPIC} "
      f"| publishes {SCENE_GRAPH_JSON_TOPIC}"
    )

  def _detections_callback(self, msg: String) -> None:
    self._latest_detections_json = msg.data

  def _detection_complete_callback(self, msg: Bool) -> None:
    if not msg.data:
      return
    self.build_and_publish()

  def build_and_publish(self) -> None:
    if self._latest_detections_json is None:
      self.get_logger().error(f"No detections received on {DETECTIONS_JSON_TOPIC}")
      return

    try:
      detections = detections_from_json(self._latest_detections_json)
      result = self._builder.build_from_detections(detections)
    except Exception as exc:
      self.get_logger().error(
        f"Scene graph build failed: {exc}\n{traceback.format_exc()}"
      )
      return

    graph_msg = String()
    graph_msg.data = scene_graph_to_json_string(result.scene)
    self._graph_pub.publish(graph_msg)

    if self.get_parameter("save_graph").get_parameter_value().bool_value:
      path = self.get_parameter("graph_output_path").get_parameter_value().string_value
      save_scene_graph_json(result.scene, path)
      self.get_logger().info(f"Saved scene graph → {path}")

    complete = Bool()
    complete.data = True
    self._complete_pub.publish(complete)
    self._graph_done = True

    self.get_logger().info(
      f"Published live scene graph | objects={result.num_objects} "
      f"relations={result.num_relations}"
    )

    if self.get_parameter("shutdown_on_complete").get_parameter_value().bool_value:
      rclpy.shutdown()


def main(args=None) -> None:
  rclpy.init(args=args)
  node = LiveSceneGraphNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == "__main__":
  main()
