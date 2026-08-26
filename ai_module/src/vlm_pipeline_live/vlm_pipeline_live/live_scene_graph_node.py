"""ROS node: build live scene graph after detection completes."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Bool, String

from vlm_pipeline_live.capture_paths import resolve_unique_graph_path
from vlm_pipeline_live.live_detector import detections_from_json
from vlm_pipeline_live.live_scene_graph import (
  LiveSceneGraphBuilder,
  save_scene_graph_json,
  scene_graph_to_json_string,
)
from vlm_pipeline_live.process_handshake import (
  DETECTION_COMPLETE as HS_DETECTION_COMPLETE,
  DETECTIONS_JSON as HS_DETECTIONS_JSON,
  SCENE_GRAPH_COMPLETE as HS_SCENE_GRAPH_COMPLETE,
  read_text,
  read_token,
  write_token,
)

DETECTIONS_JSON_TOPIC = "/vlm_live/detections_json"
DETECTION_COMPLETE_TOPIC = "/vlm_live/detection_complete"
SCENE_GRAPH_JSON_TOPIC = "/vlm_live/scene_graph_json"
SCENE_GRAPH_COMPLETE_TOPIC = "/vlm_live/scene_graph_complete"

OUTPUT_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
LIVE_QOS = qos_profile_sensor_data


class LiveSceneGraphNode(Node):
  """Subscribe to fused detections and publish a VLA-3D-format scene graph."""

  def __init__(self) -> None:
    super().__init__("vlm_live_scene_graph")

    self.declare_parameter("scene_name", "live_scene")
    self.declare_parameter("region_name", "live_map")
    self.declare_parameter("near_distance_m", 1.5)
    self.declare_parameter("auto_run_on_detection_complete", True)
    self.declare_parameter("save_graph", True)
    # Empty → unique dir under graph_output_dir. Non-empty *.json → stamped filename.
    self.declare_parameter("graph_output_path", "")
    self.declare_parameter("graph_output_dir", "/tmp/vlm_live_captures")
    self.declare_parameter("unique_graph_output", True)
    self.declare_parameter("shutdown_on_complete", False)

    self._latest_detections_json: str | None = None
    self._graph_done = False
    self._last_saved_path: str | None = None
    self._file_seen_complete = read_token(HS_DETECTION_COMPLETE)

    self._builder = LiveSceneGraphBuilder(
      scene_name=self.get_parameter("scene_name").get_parameter_value().string_value,
      region_name=self.get_parameter("region_name").get_parameter_value().string_value,
      near_distance_m=self.get_parameter("near_distance_m").get_parameter_value().double_value,
    )

    self._graph_pub = self.create_publisher(String, SCENE_GRAPH_JSON_TOPIC, LIVE_QOS)
    self._complete_pub = self.create_publisher(Bool, SCENE_GRAPH_COMPLETE_TOPIC, LIVE_QOS)

    self.create_subscription(String, DETECTIONS_JSON_TOPIC, self._detections_callback, LIVE_QOS)
    if self.get_parameter("auto_run_on_detection_complete").get_parameter_value().bool_value:
      self.create_subscription(
        Bool,
        DETECTION_COMPLETE_TOPIC,
        self._detection_complete_callback,
        LIVE_QOS,
      )
    self.create_timer(0.2, self._poll_file_complete)

    scene = self.get_parameter("scene_name").get_parameter_value().string_value
    out_dir = self.get_parameter("graph_output_dir").get_parameter_value().string_value
    self.get_logger().info(
      f"Live scene graph ready | waits for {DETECTION_COMPLETE_TOPIC} "
      f"| publishes {SCENE_GRAPH_JSON_TOPIC} "
      f"| unique saves under {out_dir}/{scene}/<timestamp>/"
    )

  def _detections_callback(self, msg: String) -> None:
    self._latest_detections_json = msg.data

  def _detection_complete_callback(self, msg: Bool) -> None:
    if not msg.data:
      return
    self.build_and_publish()

  def _poll_file_complete(self) -> None:
    token = read_token(HS_DETECTION_COMPLETE)
    if token <= self._file_seen_complete:
      return
    payload = read_text(HS_DETECTIONS_JSON)
    if payload:
      self._latest_detections_json = payload
    self._file_seen_complete = token
    self.build_and_publish()

  def _resolve_save_path(self) -> Path:
    scene_name = self.get_parameter("scene_name").get_parameter_value().string_value
    raw_path = self.get_parameter("graph_output_path").get_parameter_value().string_value
    out_dir = self.get_parameter("graph_output_dir").get_parameter_value().string_value
    unique = self.get_parameter("unique_graph_output").get_parameter_value().bool_value
    if unique:
      return resolve_unique_graph_path(
        scene_name=scene_name,
        graph_output_path=raw_path,
        graph_output_dir=out_dir,
      )
    if raw_path.strip():
      path = Path(raw_path).expanduser()
      path.parent.mkdir(parents=True, exist_ok=True)
      return path
    path = Path(out_dir).expanduser() / scene_name / "scene_graph.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

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
      path = self._resolve_save_path()
      save_scene_graph_json(result.scene, str(path))
      self._last_saved_path = str(path)
      # Convenience pointer for tools that still look for a single "latest" file.
      latest = path.parent.parent / "latest_scene_graph.json"
      try:
        latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
      except OSError:
        latest = None
      meta = {
        "scene_name": self.get_parameter("scene_name").get_parameter_value().string_value,
        "graph_path": str(path),
        "latest_pointer": str(latest) if latest is not None else "",
        "num_objects": result.num_objects,
        "num_relations": result.num_relations,
      }
      meta_path = path.parent / "capture_meta.json"
      try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
      except OSError:
        pass
      self.get_logger().info(f"Saved scene graph → {path}")
      if latest is not None:
        self.get_logger().info(f"Also updated latest pointer → {latest}")

    complete = Bool()
    complete.data = True
    self._complete_pub.publish(complete)
    write_token(HS_SCENE_GRAPH_COMPLETE, max(self._file_seen_complete, 1))
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
