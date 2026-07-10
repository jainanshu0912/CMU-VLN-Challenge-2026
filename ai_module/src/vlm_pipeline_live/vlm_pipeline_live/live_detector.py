"""GroundingDINO + LiDAR fusion for Pipeline B live object detection."""

from __future__ import annotations

import json
import os
import time
import traceback
from collections.abc import Callable
from typing import Iterable

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray

from vlm_pipeline_live.equirect_to_perspective import (
  EquirectPerspectiveProjector,
  ros_image_to_numpy,
)
from vlm_pipeline_live.grounding_dino_backend import (
  CPU_TEST_PROMPT,
  DEFAULT_INDOOR_PROMPT,
  Detection2D,
  GroundingDinoBackend,
  prompt_from_question,
)

from vlm_pipeline_live.lidar_camera_fusion import (
  DetectedObject3D,
  LidarCameraFusion,
  nms_3d,
)

LogFn = Callable[[str], None]

CAMERA_TOPIC = "/camera/image"
REGISTERED_SCAN_TOPIC = "/registered_scan"
STATE_ESTIMATION_TOPIC = "/state_estimation"
EXPLORATION_COMPLETE_TOPIC = "/vlm_live/exploration_complete"
RUN_DETECTION_TOPIC = "/vlm_live/run_detection"
CLEAR_DETECTIONS_TOPIC = "/vlm_live/clear_detections"
DETECTIONS_JSON_TOPIC = "/vlm_live/detections_json"
DETECTION_MARKERS_TOPIC = "/vlm_live/detection_markers"
DETECTION_COMPLETE_TOPIC = "/vlm_live/detection_complete"

DEFAULT_MODEL_CONFIG = os.environ.get(
  "GROUNDINGDINO_CONFIG",
  "/home/docker/models/GroundingDINO_SwinT_OGC.py",
)
DEFAULT_MODEL_CHECKPOINT = os.environ.get(
  "GROUNDINGDINO_CHECKPOINT",
  "/home/docker/models/groundingdino_swint_ogc.pth",
)

OUTPUT_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


class LiveDetector:
  """Run GroundingDINO on equirect crops and fuse detections with registered scan."""

  def __init__(
    self,
    detector: GroundingDinoBackend | None = None,
    projector: EquirectPerspectiveProjector | None = None,
    fusion: LidarCameraFusion | None = None,
    default_prompt: str = DEFAULT_INDOOR_PROMPT,
    log_fn: LogFn | None = None,
  ) -> None:
    self.projector = projector or EquirectPerspectiveProjector()
    self.detector = detector or GroundingDinoBackend(
      config_path=DEFAULT_MODEL_CONFIG,
      checkpoint_path=DEFAULT_MODEL_CHECKPOINT,
    )
    self.fusion = fusion or LidarCameraFusion(projector=self.projector)
    self.default_prompt = default_prompt
    self._log = log_fn or (lambda _msg: None)

  def detect_scene(
    self,
    equirect_rgb: np.ndarray,
    registered_scan: PointCloud2,
    odometry: Odometry,
    prompt: str | None = None,
  ) -> list[DetectedObject3D]:
    caption = prompt or self.default_prompt
    crops = self.projector.crop_all(equirect_rgb)

    detections_2d: list[Detection2D] = []
    if self.detector.is_available:
      self._log(
        f"GroundingDINO on {self.detector.device} | {len(crops)} crops | "
        f"prompt tokens={len(caption.split('.'))}"
      )
      for index, crop in enumerate(crops):
        crop_start = time.monotonic()
        crop_dets = self.detector.detect(crop.image, caption, crop.heading_deg)
        detections_2d.extend(crop_dets)
        self._log(
          f"Crop {index + 1}/{len(crops)} heading={crop.heading_deg:.0f}° "
          f"→ {len(crop_dets)} boxes in {time.monotonic() - crop_start:.1f}s"
        )
    else:
      raise RuntimeError(
        "GroundingDINO is unavailable. Install torch + GroundingDINO and provide "
        "model_config_path / model_checkpoint_path."
      )

    return self.fusion.fuse(detections_2d, registered_scan, odometry)

  def detect_scene_from_question(
    self,
    equirect_rgb: np.ndarray,
    registered_scan: PointCloud2,
    odometry: Odometry,
    question: str,
  ) -> list[DetectedObject3D]:
    return self.detect_scene(
      equirect_rgb,
      registered_scan,
      odometry,
      prompt=prompt_from_question(question),
    )


def detections_to_json(objects: Iterable[DetectedObject3D]) -> str:
  payload = {"objects": [obj.to_dict() for obj in objects]}
  return json.dumps(payload)


def detections_from_json(payload: str) -> list[DetectedObject3D]:
  data = json.loads(payload)
  return [DetectedObject3D(**item) for item in data.get("objects", [])]


def detections_to_markers(objects: Iterable[DetectedObject3D], stamp, frame_id: str = "map"):
  marker_array = MarkerArray()
  for index, obj in enumerate(objects):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = obj.label
    marker.id = index
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = obj.cx
    marker.pose.position.y = obj.cy
    marker.pose.position.z = obj.cz
    marker.pose.orientation.w = 1.0
    marker.scale.x = max(obj.x_length, 0.1)
    marker.scale.y = max(obj.y_length, 0.1)
    marker.scale.z = max(obj.z_length, 0.1)
    marker.color.a = 0.45
    marker.color.r = 0.1
    marker.color.g = 0.8
    marker.color.b = 0.2
    marker_array.markers.append(marker)
  return marker_array


class LiveDetectorNode(Node):
  """Run live detection after exploration using the latest camera image and scan."""

  def __init__(self) -> None:
    super().__init__("vlm_live_detector")

    self.declare_parameter("model_config_path", DEFAULT_MODEL_CONFIG)
    self.declare_parameter("model_checkpoint_path", DEFAULT_MODEL_CHECKPOINT)
    self.declare_parameter("box_threshold", 0.3)
    self.declare_parameter("text_threshold", 0.25)
    self.declare_parameter("detection_prompt", DEFAULT_INDOOR_PROMPT)
    self.declare_parameter("min_lidar_points", 8)
    self.declare_parameter("nms_distance_m", 0.5)
    self.declare_parameter("max_robot_distance_m", 12.0)
    self.declare_parameter("auto_run_on_exploration_complete", True)
    self.declare_parameter("allow_repeat_detection", True)
    self.declare_parameter("accumulate_detections", True)
    self.declare_parameter("save_snapshots", False)
    self.declare_parameter("snapshot_dir", "/tmp/vlm_live_snapshots")
    self.declare_parameter("shutdown_on_complete", False)
    self.declare_parameter("force_cpu", False)

    config_path = self.get_parameter("model_config_path").get_parameter_value().string_value
    checkpoint_path = (
      self.get_parameter("model_checkpoint_path").get_parameter_value().string_value
    )
    default_prompt = self.get_parameter("detection_prompt").get_parameter_value().string_value
    force_cpu = self.get_parameter("force_cpu").get_parameter_value().bool_value

    detector = GroundingDinoBackend(
      config_path=config_path,
      checkpoint_path=checkpoint_path,
      box_threshold=self.get_parameter("box_threshold").get_parameter_value().double_value,
      text_threshold=self.get_parameter("text_threshold").get_parameter_value().double_value,
      force_cpu=force_cpu,
    )
    projector = EquirectPerspectiveProjector()
    fusion = LidarCameraFusion(
      projector=projector,
      min_lidar_points=int(
        self.get_parameter("min_lidar_points").get_parameter_value().integer_value
      ),
      nms_distance_m=self.get_parameter("nms_distance_m").get_parameter_value().double_value,
      max_robot_distance_m=(
        self.get_parameter("max_robot_distance_m").get_parameter_value().double_value
      ),
    )
    self._detector = LiveDetector(
      detector=detector,
      projector=projector,
      fusion=fusion,
      default_prompt=default_prompt,
      log_fn=self.get_logger().info,
    )

    self._latest_image: Image | None = None
    self._latest_scan: PointCloud2 | None = None
    self._latest_odom: Odometry | None = None
    self._detection_done = False
    self._accumulated_objects: list[DetectedObject3D] = []
    self._snapshot_index = 0
    self._allow_repeat = (
      self.get_parameter("allow_repeat_detection").get_parameter_value().bool_value
    )
    self._accumulate = (
      self.get_parameter("accumulate_detections").get_parameter_value().bool_value
    )
    self._save_snapshots = (
      self.get_parameter("save_snapshots").get_parameter_value().bool_value
    )
    self._snapshot_dir = (
      self.get_parameter("snapshot_dir").get_parameter_value().string_value
    )
    self._nms_distance_m = (
      self.get_parameter("nms_distance_m").get_parameter_value().double_value
    )

    self._json_pub = self.create_publisher(String, DETECTIONS_JSON_TOPIC, OUTPUT_QOS)
    self._marker_pub = self.create_publisher(MarkerArray, DETECTION_MARKERS_TOPIC, OUTPUT_QOS)
    self._complete_pub = self.create_publisher(Bool, DETECTION_COMPLETE_TOPIC, OUTPUT_QOS)

    self.create_subscription(Image, CAMERA_TOPIC, self._image_callback, qos_profile_sensor_data)
    self.create_subscription(
      PointCloud2,
      REGISTERED_SCAN_TOPIC,
      self._scan_callback,
      qos_profile_sensor_data,
    )
    self.create_subscription(
      Odometry,
      STATE_ESTIMATION_TOPIC,
      self._odom_callback,
      qos_profile_sensor_data,
    )

    if self.get_parameter("auto_run_on_exploration_complete").get_parameter_value().bool_value:
      self.create_subscription(
        Bool,
        EXPLORATION_COMPLETE_TOPIC,
        self._exploration_complete_callback,
        OUTPUT_QOS,
      )
    else:
      self.create_subscription(Bool, RUN_DETECTION_TOPIC, self._run_detection_callback, OUTPUT_QOS)
      self.create_subscription(Bool, CLEAR_DETECTIONS_TOPIC, self._clear_detections_callback, OUTPUT_QOS)

    if self._save_snapshots:
      os.makedirs(self._snapshot_dir, exist_ok=True)

    if detector.is_available:
      self.get_logger().info(
        f"GroundingDINO backend available | device={detector.device}"
      )
      if detector.device == "cpu":
        self.get_logger().warn(
          "Running on CPU — first detection may take several minutes for 4 crops."
        )
    else:
      self.get_logger().warn(
        "GroundingDINO not installed or torch missing. "
        "Detection will fail until models/deps are installed."
      )

    self.get_logger().info(
      f"Live detector ready | prompt_len={len(default_prompt)} "
      f"| config={config_path}"
    )
    if not self.get_parameter("auto_run_on_exploration_complete").get_parameter_value().bool_value:
      self.get_logger().info(
        f"Manual mode — teleop the robot, then trigger detection:\n"
        f"  ros2 topic pub --once {RUN_DETECTION_TOPIC} std_msgs/msg/Bool '{{data: true}}'\n"
        f"  ros2 topic pub --once {CLEAR_DETECTIONS_TOPIC} std_msgs/msg/Bool '{{data: true}}'"
      )
      if self._accumulate:
        self.get_logger().info("Accumulating detections across triggers (NMS merge).")
      if self._save_snapshots:
        self.get_logger().info(f"Saving snapshots to {self._snapshot_dir}")

  def _image_callback(self, msg: Image) -> None:
    self._latest_image = msg

  def _scan_callback(self, msg: PointCloud2) -> None:
    self._latest_scan = msg

  def _odom_callback(self, msg: Odometry) -> None:
    self._latest_odom = msg

  def _exploration_complete_callback(self, msg: Bool) -> None:
    if not msg.data or (self._detection_done and not self._allow_repeat):
      return
    self.run_detection()

  def _run_detection_callback(self, msg: Bool) -> None:
    if not msg.data:
      return
    if self._detection_done and not self._allow_repeat:
      self.get_logger().warn("Detection already ran; set allow_repeat_detection:=true to re-run.")
      return
    self.run_detection()

  def _clear_detections_callback(self, msg: Bool) -> None:
    if not msg.data:
      return
    self._accumulated_objects = []
    self._detection_done = False
    self._snapshot_index = 0
    self.get_logger().info("Cleared accumulated detection map.")

  def _save_snapshot(self, equirect: np.ndarray, objects: list[DetectedObject3D]) -> None:
    if not self._save_snapshots or self._latest_odom is None:
      return

    stamp = int(time.time())
    index = self._snapshot_index
    self._snapshot_index += 1
    base = os.path.join(self._snapshot_dir, f"snapshot_{index:04d}_{stamp}")

    try:
      from PIL import Image as PILImage

      PILImage.fromarray(equirect).save(f"{base}_equirect.png")
    except ImportError:
      self.get_logger().warn("PIL unavailable — skipping equirect snapshot image.")

    pose = self._latest_odom.pose.pose
    meta = {
      "index": index,
      "timestamp": stamp,
      "position": {
        "x": pose.position.x,
        "y": pose.position.y,
        "z": pose.position.z,
      },
      "orientation": {
        "x": pose.orientation.x,
        "y": pose.orientation.y,
        "z": pose.orientation.z,
        "w": pose.orientation.w,
      },
      "num_detections": len(objects),
      "objects": [obj.to_dict() for obj in objects],
    }
    with open(f"{base}_meta.json", "w", encoding="utf-8") as handle:
      json.dump(meta, handle, indent=2)

    self.get_logger().info(f"Saved snapshot {index} → {base}_*")

  def run_detection(self, prompt: str | None = None) -> list[DetectedObject3D]:
    if self._latest_image is None:
      self.get_logger().error(f"No image received on {CAMERA_TOPIC}")
      return []
    if self._latest_scan is None:
      self.get_logger().error(f"No scan received on {REGISTERED_SCAN_TOPIC}")
      return []
    if self._latest_odom is None:
      self.get_logger().error(f"No odometry received on {STATE_ESTIMATION_TOPIC}")
      return []

    try:
      detection_start = time.monotonic()
      pose = self._latest_odom.pose.pose
      self.get_logger().info(
        "Starting live detection (GroundingDINO + LiDAR fusion) at "
        f"({pose.position.x:.2f}, {pose.position.y:.2f}, "
        f"z={pose.position.z:.2f})..."
      )
      equirect = ros_image_to_numpy(self._latest_image)
      objects = self._detector.detect_scene(
        equirect,
        self._latest_scan,
        self._latest_odom,
        prompt=prompt,
      )
    except Exception as exc:
      self.get_logger().error(
        f"Live detection failed: {exc}\n{traceback.format_exc()}"
      )
      return []

    if self._accumulate:
      self._accumulated_objects = nms_3d(
        self._accumulated_objects + objects,
        self._nms_distance_m,
      )
      publish_objects = self._accumulated_objects
      self.get_logger().info(
        f"Snapshot added {len(objects)} objects "
        f"(map total {len(publish_objects)} after NMS)."
      )
    else:
      publish_objects = objects

    self._save_snapshot(equirect, objects)
    self._publish_results(publish_objects)
    self._detection_done = True
    elapsed = time.monotonic() - detection_start
    self.get_logger().info(
      f"Published {len(publish_objects)} fused 3D detections in {elapsed:.1f}s"
    )

    if self.get_parameter("shutdown_on_complete").get_parameter_value().bool_value:
      rclpy.shutdown()
    return publish_objects

  def _publish_results(self, objects: list[DetectedObject3D]) -> None:
    json_msg = String()
    json_msg.data = detections_to_json(objects)
    self._json_pub.publish(json_msg)

    stamp = self.get_clock().now().to_msg()
    self._marker_pub.publish(detections_to_markers(objects, stamp))

    complete = Bool()
    complete.data = True
    self._complete_pub.publish(complete)


def main(args=None) -> None:
  rclpy.init(args=args)
  node = LiveDetectorNode()
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
