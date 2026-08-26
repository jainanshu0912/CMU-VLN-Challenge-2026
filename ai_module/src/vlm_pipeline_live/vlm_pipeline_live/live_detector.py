"""GroundingDINO + LiDAR fusion for Pipeline B live object detection."""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray

from vlm_pipeline_live.detection_backend import (
  Detection2D,
  create_detection_backend,
)
from vlm_pipeline_live.detection_vis import (
  draw_detections_on_image,
  draw_projected_points,
)
from vlm_pipeline_live.equirect_to_perspective import (
  EquirectPerspectiveProjector,
  ros_image_to_numpy,
)
from vlm_pipeline_live.grounding_dino_backend import (
  DEFAULT_INDOOR_PROMPT,
  prompt_for_scene_type,
  prompt_from_question,
)

from vlm_pipeline_live.lidar_camera_fusion import (
  DetectedObject3D,
  LidarCameraFusion,
  map_points_to_camera,
  nms_3d,
)
from vlm_pipeline_live.process_handshake import (
  DETECTION_COMPLETE as HS_DETECTION_COMPLETE,
  DETECTIONS_JSON as HS_DETECTIONS_JSON,
  RUN_DETECTION as HS_RUN_DETECTION,
  read_token,
  write_text,
  write_token,
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
# Same profile as /camera/image — RELIABLE Bool topics do not match in this container.
LIVE_QOS = qos_profile_sensor_data


@dataclass
class CropDebugImage:
  """One perspective crop with DINO boxes (+ optional fused-center overlay)."""

  heading_deg: float
  image_rgb: np.ndarray
  detections: list[Detection2D]
  annotated_rgb: np.ndarray


@dataclass
class DetectSceneResult:
  objects: list[DetectedObject3D]
  detections_2d: list[Detection2D] = field(default_factory=list)
  crop_debug: list[CropDebugImage] = field(default_factory=list)


class LiveDetector:
  """Run an open-vocab 2D backend on equirect crops and fuse with LiDAR."""

  def __init__(
    self,
    detector=None,
    projector: EquirectPerspectiveProjector | None = None,
    fusion: LidarCameraFusion | None = None,
    verifier: Any | None = None,
    default_prompt: str = DEFAULT_INDOOR_PROMPT,
    log_fn: LogFn | None = None,
  ) -> None:
    self.projector = projector or EquirectPerspectiveProjector()
    self.detector = detector or create_detection_backend(
      "grounding_dino",
      device="cuda",
      model_config_path=DEFAULT_MODEL_CONFIG,
      model_checkpoint_path=DEFAULT_MODEL_CHECKPOINT,
    )
    self.fusion = fusion or LidarCameraFusion(projector=self.projector)
    self.verifier = verifier
    self.default_prompt = default_prompt
    self._log = log_fn or (lambda _msg: None)

  def detect_scene(
    self,
    equirect_rgb: np.ndarray,
    registered_scan: PointCloud2,
    odometry: Odometry,
    prompt: str | None = None,
  ) -> DetectSceneResult:
    caption = prompt or self.default_prompt
    crops = self.projector.crop_all(equirect_rgb)

    detections_2d: list[Detection2D] = []
    crop_debug: list[CropDebugImage] = []
    if self.detector.is_available:
      self._log(
        f"{self.detector.name} on {self.detector.device} | {len(crops)} crops | "
        f"prompt tokens={len(caption.split('.'))}"
        + (
          f" | gemini_verify={self.verifier.model}"
          if self.verifier is not None and self.verifier.enabled
          else ""
        )
      )
      for index, crop in enumerate(crops):
        crop_start = time.monotonic()
        crop_dets = self.detector.detect(crop.image, caption, crop.heading_deg)
        if self.verifier is not None and self.verifier.enabled:
          crop_dets = self.verifier.verify_crop(crop.image, crop_dets, caption)
        detections_2d.extend(crop_dets)
        annotated = draw_detections_on_image(crop.image, crop_dets)
        crop_debug.append(
          CropDebugImage(
            heading_deg=float(crop.heading_deg),
            image_rgb=np.asarray(crop.image, dtype=np.uint8).copy(),
            detections=list(crop_dets),
            annotated_rgb=annotated,
          )
        )
        self._log(
          f"Crop {index + 1}/{len(crops)} heading={crop.heading_deg:.0f}° "
          f"→ {len(crop_dets)} boxes in {time.monotonic() - crop_start:.1f}s"
        )
    else:
      raise RuntimeError(
        f"Detector backend '{getattr(self.detector, 'name', '?')}' is unavailable. "
        "Install its dependencies and provide model paths/weights."
      )

    objects = self.fusion.fuse(detections_2d, registered_scan, odometry)
    crop_debug = self._overlay_fused_centers(crop_debug, objects, odometry)
    return DetectSceneResult(
      objects=objects,
      detections_2d=detections_2d,
      crop_debug=crop_debug,
    )

  def _overlay_fused_centers(
    self,
    crop_debug: list[CropDebugImage],
    objects: list[DetectedObject3D],
    odometry: Odometry,
  ) -> list[CropDebugImage]:
    """Draw cyan crosses where fused 3D object centers project into each crop.

    If the cyan mark sits far outside the green DINO box, LiDAR/camera geometry
    is likely misaligned (or the box matched the wrong points).
    """
    if not objects or not crop_debug:
      return crop_debug

    centers = np.array([[o.cx, o.cy, o.cz] for o in objects], dtype=np.float64)
    labels = [o.label for o in objects]
    points_cam = map_points_to_camera(centers, odometry)

    updated: list[CropDebugImage] = []
    for crop in crop_debug:
      px, py, visible = self.projector.project_camera_points_to_crop(
        points_cam,
        crop.heading_deg,
      )
      # Prefer centers that came from this crop's heading when available.
      same_view = np.array(
        [
          abs(float(o.source_heading_deg) - float(crop.heading_deg)) <= 0.5
          for o in objects
        ],
        dtype=bool,
      )
      mask = visible & same_view
      if not np.any(mask):
        mask = visible
      annotated = draw_projected_points(
        crop.annotated_rgb,
        px[mask],
        py[mask],
        np.ones(int(np.count_nonzero(mask)), dtype=bool),
        color=(0, 255, 255),
        radius=5,
        labels=[labels[i] for i, ok in enumerate(mask) if ok],
      )
      updated.append(
        CropDebugImage(
          heading_deg=crop.heading_deg,
          image_rgb=crop.image_rgb,
          detections=crop.detections,
          annotated_rgb=annotated,
        )
      )
    return updated

  def detect_scene_from_question(
    self,
    equirect_rgb: np.ndarray,
    registered_scan: PointCloud2,
    odometry: Odometry,
    question: str,
  ) -> DetectSceneResult:
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

    self.declare_parameter("detector_backend", "grounding_dino")
    self.declare_parameter("model_config_path", DEFAULT_MODEL_CONFIG)
    self.declare_parameter("model_checkpoint_path", DEFAULT_MODEL_CHECKPOINT)
    self.declare_parameter("yolo_model", "")
    self.declare_parameter("box_threshold", 0.35)
    self.declare_parameter("text_threshold", 0.25)
    # Empty detection_prompt → choose from scene_type (office / hotel / indoor).
    self.declare_parameter("scene_type", "office")
    self.declare_parameter("detection_prompt", "")
    self.declare_parameter("min_lidar_points", 8)
    self.declare_parameter("nms_distance_m", 0.5)
    self.declare_parameter("max_robot_distance_m", 12.0)
    self.declare_parameter("auto_run_on_exploration_complete", True)
    self.declare_parameter("allow_repeat_detection", True)
    self.declare_parameter("accumulate_detections", True)
    self.declare_parameter("save_snapshots", False)
    self.declare_parameter("snapshot_dir", "/tmp/vlm_live_snapshots")
    self.declare_parameter("shutdown_on_complete", False)
    self.declare_parameter("device", "cuda")
    self.declare_parameter("gemini_verify", False)
    self.declare_parameter("gemini_model", "gemini-3.6-flash")
    self.declare_parameter("gemini_api_key", "")
    self.declare_parameter("gemini_fail_open", True)

    backend_name = (
      self.get_parameter("detector_backend").get_parameter_value().string_value.strip()
      or "grounding_dino"
    )
    config_path = self.get_parameter("model_config_path").get_parameter_value().string_value
    checkpoint_path = (
      self.get_parameter("model_checkpoint_path").get_parameter_value().string_value
    )
    yolo_model = self.get_parameter("yolo_model").get_parameter_value().string_value.strip()
    scene_type = self.get_parameter("scene_type").get_parameter_value().string_value
    prompt_override = (
      self.get_parameter("detection_prompt").get_parameter_value().string_value.strip()
    )
    default_prompt = prompt_override or prompt_for_scene_type(scene_type)
    device = self.get_parameter("device").get_parameter_value().string_value.strip() or "cuda"

    detector = create_detection_backend(
      backend_name,
      device=device,
      box_threshold=self.get_parameter("box_threshold").get_parameter_value().double_value,
      text_threshold=self.get_parameter("text_threshold").get_parameter_value().double_value,
      model_config_path=config_path,
      model_checkpoint_path=checkpoint_path,
      yolo_model=yolo_model,
    )
    self.get_logger().info(
      f"2D detector backend={detector.name} device={detector.device}"
    )

    gemini_verify = self.get_parameter("gemini_verify").get_parameter_value().bool_value
    gemini_model = (
      self.get_parameter("gemini_model").get_parameter_value().string_value.strip()
      or "gemini-3.6-flash"
    )
    gemini_api_key = (
      self.get_parameter("gemini_api_key").get_parameter_value().string_value.strip()
    )
    gemini_fail_open = self.get_parameter("gemini_fail_open").get_parameter_value().bool_value
    verifier: Any | None = None
    if gemini_verify:
      from vlm_pipeline_live.gemini_label_verifier import GeminiLabelVerifier

      verifier = GeminiLabelVerifier(
        model=gemini_model,
        api_key=gemini_api_key or None,
        enabled=True,
        fail_open=gemini_fail_open,
        log_fn=self.get_logger().info,
      )
      if verifier.is_available:
        self.get_logger().info(
          f"Gemini label verify ON model={verifier.model} "
          f"key={verifier.api_key_fingerprint}"
        )
      else:
        self.get_logger().warn(
          "gemini_verify:=true but Gemini is unavailable "
          "(set a valid GEMINI_API_KEY from https://aistudio.google.com/apikey, "
          "recreate ai_module so the env is visible, and "
          "pip3 install -U --break-system-packages google-genai pillow). "
          "Will fail-open and keep detector labels."
          if gemini_fail_open
          else "gemini_verify:=true but Gemini is unavailable — detection will error."
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
      verifier=verifier,
      default_prompt=default_prompt,
      log_fn=self.get_logger().info,
    )

    self._latest_image: Image | None = None
    self._latest_scan: PointCloud2 | None = None
    self._latest_odom: Odometry | None = None
    self._got_image = False
    self._got_scan = False
    self._got_odom = False
    self._detection_done = False
    self._detect_lock = threading.Lock()
    self._sensor_wait_sec = 15.0
    self._file_seen_token = read_token(HS_RUN_DETECTION)
    self._active_token = 0
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

    self._json_pub = self.create_publisher(String, DETECTIONS_JSON_TOPIC, LIVE_QOS)
    self._marker_pub = self.create_publisher(MarkerArray, DETECTION_MARKERS_TOPIC, OUTPUT_QOS)
    self._complete_pub = self.create_publisher(Bool, DETECTION_COMPLETE_TOPIC, LIVE_QOS)

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

    # Explorer per-stop triggers. BEST_EFFORT matches camera; file handshake is backup.
    self.create_subscription(
      Bool, RUN_DETECTION_TOPIC, self._run_detection_callback, LIVE_QOS
    )
    self.create_subscription(Bool, CLEAR_DETECTIONS_TOPIC, self._clear_detections_callback, LIVE_QOS)
    if self.get_parameter("auto_run_on_exploration_complete").get_parameter_value().bool_value:
      self.create_subscription(
        Bool,
        EXPLORATION_COMPLETE_TOPIC,
        self._exploration_complete_callback,
        LIVE_QOS,
      )
    self.create_timer(0.2, self._poll_file_trigger)

    if self._save_snapshots:
      os.makedirs(self._snapshot_dir, exist_ok=True)

    if detector.is_available:
      self.get_logger().info(
        f"GroundingDINO backend available | device={detector.device}"
      )
      self.get_logger().info(
        "Running on GPU — expect roughly a few seconds per crop after model load."
      )
    else:
      self.get_logger().warn(
        "GroundingDINO not installed or torch missing. "
        "Detection will fail until models/deps are installed."
      )

    self.get_logger().info(
      f"Live detector ready | scene_type={scene_type} "
      f"| prompt_tokens={len([t for t in default_prompt.split('.') if t.strip()])} "
      f"| class-aware NMS + label canonicalize | config={config_path}"
    )
    self.get_logger().info(
      f"Detection trigger: {RUN_DETECTION_TOPIC} "
      f"(explorer per-stop or manual teleop)."
    )
    if self.get_parameter("auto_run_on_exploration_complete").get_parameter_value().bool_value:
      self.get_logger().info(
        f"Also running once on {EXPLORATION_COMPLETE_TOPIC}=true"
      )
    else:
      self.get_logger().info(
        f"Manual / explorer-stop mode — also:\n"
        f"  ros2 topic pub --once {RUN_DETECTION_TOPIC} std_msgs/msg/Bool '{{data: true}}'\n"
        f"  ros2 topic pub --once {CLEAR_DETECTIONS_TOPIC} std_msgs/msg/Bool '{{data: true}}'"
      )
    if self._accumulate:
      self.get_logger().info("Accumulating detections across triggers (NMS merge).")
    if self._save_snapshots:
      self.get_logger().info(f"Saving snapshots to {self._snapshot_dir}")

    if detector.is_available:
      threading.Thread(target=self._preload_model, daemon=True).start()

  def _preload_model(self) -> None:
    loader = getattr(self._detector.detector, "_load_model", None)
    if loader is None:
      return
    try:
      self.get_logger().info("Preloading 2D detector weights in the background...")
      loader()
      self.get_logger().info("2D detector weights ready.")
    except Exception as exc:
      self.get_logger().error(f"Detector preload failed: {exc}\n{traceback.format_exc()}")

  def _image_callback(self, msg: Image) -> None:
    self._latest_image = msg
    if not self._got_image:
      self._got_image = True
      self.get_logger().info(
        f"Receiving {CAMERA_TOPIC} ({msg.width}x{msg.height})"
      )

  def _scan_callback(self, msg: PointCloud2) -> None:
    self._latest_scan = msg
    if not self._got_scan:
      self._got_scan = True
      self.get_logger().info(f"Receiving {REGISTERED_SCAN_TOPIC}")

  def _odom_callback(self, msg: Odometry) -> None:
    self._latest_odom = msg
    if not self._got_odom:
      self._got_odom = True
      pose = msg.pose.pose.position
      self.get_logger().info(
        f"Receiving {STATE_ESTIMATION_TOPIC} at ({pose.x:.2f}, {pose.y:.2f})"
      )

  def _exploration_complete_callback(self, msg: Bool) -> None:
    if not msg.data or (self._detection_done and not self._allow_repeat):
      return
    self._start_detection_async("exploration_complete")

  def _run_detection_callback(self, msg: Bool) -> None:
    if not msg.data:
      return
    if self._detection_done and not self._allow_repeat:
      self.get_logger().warn("Detection already ran; set allow_repeat_detection:=true to re-run.")
      return
    self._start_detection_async("run_detection")

  def _poll_file_trigger(self) -> None:
    req = read_token(HS_RUN_DETECTION)
    if req <= self._file_seen_token:
      return
    self._file_seen_token = req
    self._active_token = req
    if self._detection_done and not self._allow_repeat:
      return
    self._start_detection_async("file")

  def _start_detection_async(self, source: str) -> None:
    print(
      f"[vlm_live_detector] Trigger from {source} | "
      f"image={self._latest_image is not None} "
      f"scan={self._latest_scan is not None} "
      f"odom={self._latest_odom is not None}",
      flush=True,
    )
    self.get_logger().info(
      f"Received detection trigger ({source}) | "
      f"image={self._latest_image is not None} "
      f"scan={self._latest_scan is not None} "
      f"odom={self._latest_odom is not None}"
    )
    if not self._detect_lock.acquire(blocking=False):
      self.get_logger().warn("Detection already running; ignoring extra trigger.")
      return
    threading.Thread(target=self._run_detection_worker, daemon=True).start()

  def _run_detection_worker(self) -> None:
    try:
      self.run_detection()
    finally:
      self._detect_lock.release()

  def _clear_detections_callback(self, msg: Bool) -> None:
    if not msg.data:
      return
    self._accumulated_objects = []
    self._detection_done = False
    self._snapshot_index = 0
    self.get_logger().info("Cleared accumulated detection map.")

  def _save_snapshot(
    self,
    equirect: np.ndarray,
    objects: list[DetectedObject3D],
    crop_debug: list[CropDebugImage] | None = None,
    detections_2d: list[Detection2D] | None = None,
  ) -> None:
    if not self._save_snapshots or self._latest_odom is None:
      return

    stamp = int(time.time())
    index = self._snapshot_index
    self._snapshot_index += 1
    snap_dir = os.path.join(self._snapshot_dir, f"snapshot_{index:04d}_{stamp}")
    os.makedirs(snap_dir, exist_ok=True)
    base = os.path.join(snap_dir, f"snapshot_{index:04d}")

    try:
      from PIL import Image as PILImage

      PILImage.fromarray(equirect).save(f"{base}_equirect.png")

      if crop_debug:
        for crop in crop_debug:
          h = int(round(crop.heading_deg)) % 360
          # Raw perspective crop (no overlays).
          PILImage.fromarray(crop.image_rgb).save(f"{base}_crop_h{h:03d}.png")
          # Boxes + labels (+ cyan fused centers when available).
          PILImage.fromarray(crop.annotated_rgb).save(
            f"{base}_crop_h{h:03d}_boxes.png"
          )
    except ImportError:
      self.get_logger().warn("PIL unavailable — skipping snapshot images.")
    except Exception as exc:
      self.get_logger().warn(f"Failed saving crop debug images: {exc}")

    pose = self._latest_odom.pose.pose
    meta = {
      "index": index,
      "timestamp": stamp,
      "snapshot_dir": snap_dir,
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
      "num_detections_3d": len(objects),
      "num_detections_2d": len(detections_2d or []),
      "objects": [obj.to_dict() for obj in objects],
      "detections_2d": [
        {
          "label": det.label,
          "confidence": det.confidence,
          "x1": det.x1,
          "y1": det.y1,
          "x2": det.x2,
          "y2": det.y2,
          "heading_deg": det.heading_deg,
        }
        for det in (detections_2d or [])
      ],
      "crops": [
        {
          "heading_deg": crop.heading_deg,
          "num_boxes": len(crop.detections),
          "raw_image": f"snapshot_{index:04d}_crop_h{int(round(crop.heading_deg)) % 360:03d}.png",
          "boxes_image": (
            f"snapshot_{index:04d}_crop_h{int(round(crop.heading_deg)) % 360:03d}_boxes.png"
          ),
        }
        for crop in (crop_debug or [])
      ],
      "legend": {
        "green_boxes": "GroundingDINO 2D detections on 90° perspective crops",
        "cyan_crosses": (
          "Fused 3D object centers re-projected into the crop "
          "(misalignment if far outside the matching box)"
        ),
      },
    }
    with open(f"{base}_meta.json", "w", encoding="utf-8") as handle:
      json.dump(meta, handle, indent=2)

    self.get_logger().info(
      f"Saved snapshot {index} → {snap_dir}/ "
      f"({len(crop_debug or [])} annotated crops)"
    )

  def run_detection(self, prompt: str | None = None) -> list[DetectedObject3D]:
    if not self._wait_for_sensors(self._sensor_wait_sec):
      missing = []
      if self._latest_image is None:
        missing.append(CAMERA_TOPIC)
      if self._latest_scan is None:
        missing.append(REGISTERED_SCAN_TOPIC)
      if self._latest_odom is None:
        missing.append(STATE_ESTIMATION_TOPIC)
      self.get_logger().error(
        "Cannot run detection; no data yet on: " + ", ".join(missing)
      )
      self._publish_complete()
      return []

    try:
      detection_start = time.monotonic()
      pose = self._latest_odom.pose.pose
      self.get_logger().info(
        "Starting live detection (GroundingDINO + LiDAR fusion) at "
        f"({pose.position.x:.2f}, {pose.position.y:.2f}, "
        f"z={pose.position.z:.2f})..."
      )
      print("[vlm_live_detector] Starting live detection...", flush=True)
      equirect = ros_image_to_numpy(self._latest_image)
      result = self._detector.detect_scene(
        equirect,
        self._latest_scan,
        self._latest_odom,
        prompt=prompt,
      )
      objects = result.objects
    except Exception as exc:
      self.get_logger().error(
        f"Live detection failed: {exc}\n{traceback.format_exc()}"
      )
      self._publish_complete()
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

    self._save_snapshot(
      equirect,
      objects,
      crop_debug=result.crop_debug,
      detections_2d=result.detections_2d,
    )
    self._publish_results(publish_objects)
    self._detection_done = True
    elapsed = time.monotonic() - detection_start
    self.get_logger().info(
      f"Published {len(publish_objects)} fused 3D detections in {elapsed:.1f}s"
    )
    print(
      f"[vlm_live_detector] Published {len(publish_objects)} detections in {elapsed:.1f}s",
      flush=True,
    )

    if self.get_parameter("shutdown_on_complete").get_parameter_value().bool_value:
      rclpy.shutdown()
    return publish_objects

  def _wait_for_sensors(self, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
      if (
        self._latest_image is not None
        and self._latest_scan is not None
        and self._latest_odom is not None
      ):
        return True
      time.sleep(0.1)
    return (
      self._latest_image is not None
      and self._latest_scan is not None
      and self._latest_odom is not None
    )

  def _publish_complete(self) -> None:
    complete = Bool()
    complete.data = True
    self._complete_pub.publish(complete)
    token = self._active_token or read_token(HS_RUN_DETECTION)
    if token > 0:
      write_token(HS_DETECTION_COMPLETE, token)

  def _publish_results(self, objects: list[DetectedObject3D]) -> None:
    json_msg = String()
    json_msg.data = detections_to_json(objects)
    self._json_pub.publish(json_msg)
    write_text(HS_DETECTIONS_JSON, json_msg.data)

    stamp = self.get_clock().now().to_msg()
    self._marker_pub.publish(detections_to_markers(objects, stamp))
    self._publish_complete()


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
