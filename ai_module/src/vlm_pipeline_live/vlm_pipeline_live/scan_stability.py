"""Track registered_scan stability for exploration gating."""

from __future__ import annotations

import time
from collections import deque

from sensor_msgs.msg import PointCloud2


def point_cloud_size(msg: PointCloud2) -> int:
  """Approximate point count from a PointCloud2 message."""
  if msg.width > 0 and msg.height > 0:
    return int(msg.width * msg.height)
  step = max(int(msg.point_step), 1)
  return len(msg.data) // step


class ScanStabilityMonitor:
  """True when scan size variation stays below a ratio for a time window."""

  def __init__(
    self,
    window_sec: float = 5.0,
    change_threshold: float = 0.05,
  ) -> None:
    self._window_sec = window_sec
    self._change_threshold = change_threshold
    self._samples: deque[tuple[float, int]] = deque()

  @property
  def change_threshold(self) -> float:
    return self._change_threshold

  def reset(self) -> None:
    self._samples.clear()

  def update(self, msg: PointCloud2, now: float | None = None) -> None:
    timestamp = now if now is not None else time.monotonic()
    self._samples.append((timestamp, point_cloud_size(msg)))
    self._trim(timestamp)

  def _trim(self, now: float) -> None:
    cutoff = now - self._window_sec
    while self._samples and self._samples[0][0] < cutoff:
      self._samples.popleft()

  @property
  def sample_count(self) -> int:
    return len(self._samples)

  def is_stable(self) -> bool:
    if len(self._samples) < 2:
      return False

    counts = [count for _, count in self._samples]
    mean_count = sum(counts) / len(counts)
    if mean_count <= 0:
      return False

    spread = max(counts) - min(counts)
    return (spread / mean_count) <= self._change_threshold

  def relative_change(self) -> float:
    if len(self._samples) < 2:
      return float("inf")

    counts = [count for _, count in self._samples]
    mean_count = sum(counts) / len(counts)
    if mean_count <= 0:
      return float("inf")
    return (max(counts) - min(counts)) / mean_count
