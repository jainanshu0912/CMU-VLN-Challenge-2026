"""Wait for the live scene graph, then answer /challenge_question with Pipeline A."""

from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Bool, String

from vlm_pipeline.main_node import VlmPipelineNode
from vlm_sequential.export_paths import is_scene_exported

EXPLORATION_COMPLETE_TOPIC = "/vlm_live/exploration_complete"
SCENE_READY_TOPIC = "/vlm_sequential/scene_ready"
QUESTION_FILE = Path("/tmp/vlm_live_handshake/challenge_question")

COMPLETE_QOS = QoSProfile(
  depth=1,
  reliability=ReliabilityPolicy.RELIABLE,
  durability=DurabilityPolicy.TRANSIENT_LOCAL,
  history=HistoryPolicy.KEEP_LAST,
)


class SequentialNode(VlmPipelineNode):
  """Pipeline A answering, delayed until live exploration exports a graph."""

  def __init__(self) -> None:
    super().__init__(node_name="vlm_sequential")

    self._export_root = str(self.scene_loader.data_root)
    self._scene_ready = False
    self._pending_question = ""
    self._exploration_complete = False

    self._scene_ready_pub = self.create_publisher(Bool, SCENE_READY_TOPIC, 10)
    self.create_subscription(
      Bool,
      EXPLORATION_COMPLETE_TOPIC,
      self._exploration_complete_callback,
      COMPLETE_QOS,
    )
    self.create_subscription(
      Bool,
      EXPLORATION_COMPLETE_TOPIC,
      self._exploration_complete_callback,
      qos_profile_sensor_data,
    )
    self.create_timer(2.0, self._poll_exported_scene)
    self.create_timer(0.5, self._poll_question_file)
    self._file_question = ""

    self.get_logger().info(
      "Sequential pipeline: explore first, then answer. "
      f"Waiting for {EXPLORATION_COMPLETE_TOPIC} and "
      f"{self._export_root}/{self.scene_name}/"
    )
    self.get_logger().info(
      "Local one-shot test (waits for a subscriber, publishes once, exits):\n"
      "  ros2 run vlm_pipeline pub_challenge_question "
      "\"How many sofas are below a window?\""
    )

  def _question_callback(self, msg: String) -> None:
    text = msg.data.strip()
    if not text:
      return
    if not self._scene_ready:
      if text != self._pending_question:
        preview = text if len(text) <= 100 else f"{text[:97]}..."
        self.get_logger().info(
          f"Buffering /challenge_question until the live graph is ready: {preview}"
        )
      self._pending_question = text
      return
    super()._question_callback(msg)

  def _exploration_complete_callback(self, msg: Bool) -> None:
    if not msg.data:
      return
    self._exploration_complete = True
    self.get_logger().info("Exploration complete — loading exported scene graph")
    self._try_load_scene()

  def _poll_exported_scene(self) -> None:
    if self._scene_ready:
      return
    if not self._exploration_complete:
      if is_scene_exported(self._export_root, self.scene_name):
        self._exploration_complete = True
        self.get_logger().info(
          "Found exported live graph on disk — loading without ROS complete topic"
        )
      else:
        return
    self._try_load_scene()

  def _try_load_scene(self) -> None:
    if self._scene_ready:
      return
    if not is_scene_exported(self._export_root, self.scene_name):
      if self._exploration_complete:
        self.get_logger().warn(
          f"Exploration finished but no CSV/JSON yet under "
          f"{self._export_root}/{self.scene_name} — still polling"
        )
      return
    if not self.load_deferred_scene(self._export_root, self.scene_name):
      return

    self._scene_ready = True
    ready = Bool()
    ready.data = True
    self._scene_ready_pub.publish(ready)
    self._answer_pending()

  def _answer_pending(self) -> None:
    text = self._pending_question
    if not text:
      self.get_logger().info("Live graph loaded — awaiting /challenge_question")
      return
    self.get_logger().info("Live graph loaded — answering buffered challenge question")
    queued = String()
    queued.data = text
    super()._question_callback(queued)

  def _poll_question_file(self) -> None:
    if not QUESTION_FILE.is_file():
      return
    try:
      text = QUESTION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
      return
    if not text or text == self._file_question:
      return
    self._file_question = text
    msg = String()
    msg.data = text
    self._question_callback(msg)


def main(args=None) -> None:
  rclpy.init(args=args)
  node = SequentialNode()
  executor = MultiThreadedExecutor(num_threads=4)
  executor.add_node(node)
  try:
    executor.spin()
  except KeyboardInterrupt:
    pass
  finally:
    executor.shutdown()
    node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == "__main__":
  main()
