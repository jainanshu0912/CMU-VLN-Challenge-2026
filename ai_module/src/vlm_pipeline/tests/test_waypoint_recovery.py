"""Unit tests for backup-then-retry stuck recovery geometry."""

from __future__ import annotations

import math
import unittest

from vlm_pipeline.waypoint_recovery import backup_waypoint


class BackupWaypointTest(unittest.TestCase):
  def test_backs_away_from_goal(self) -> None:
    # Robot at origin, goal to +Y. Backup must go to -Y.
    bx, by, theta = backup_waypoint(0.0, 0.0, 0.0, 0.0, 8.0, backup_m=1.2)
    self.assertAlmostEqual(bx, 0.0, places=5)
    self.assertLess(by, -1.0)
    self.assertAlmostEqual(math.hypot(bx, by), 1.2, places=5)
    self.assertAlmostEqual(theta, math.atan2(by, bx), places=5)

  def test_matches_explorer_stuck_pose(self) -> None:
    # Same numbers as the livingroom wedge: robot south, goal north.
    rx, ry = -1.67, -6.97
    gx, gy = 1.72, -1.81
    bx, by, _ = backup_waypoint(rx, ry, 0.0, gx, gy, backup_m=1.2)
    # Backup is farther from the goal than the robot is.
    d_robot = math.hypot(gx - rx, gy - ry)
    d_back = math.hypot(gx - bx, gy - by)
    self.assertGreater(d_back, d_robot)
    self.assertGreater(math.hypot(bx - rx, by - ry), 1.0)

  def test_later_recovery_offsets_laterally(self) -> None:
    first = backup_waypoint(0.0, 0.0, 0.0, 4.0, 0.0, backup_m=1.2, recovery_index=0)
    second = backup_waypoint(0.0, 0.0, 0.0, 4.0, 0.0, backup_m=1.2, recovery_index=1)
    self.assertAlmostEqual(first[1], 0.0, places=5)
    self.assertGreater(abs(second[1]), 0.2)
    self.assertGreater(
      math.hypot(first[0] - second[0], first[1] - second[1]),
      0.3,
    )


if __name__ == "__main__":
  unittest.main()
