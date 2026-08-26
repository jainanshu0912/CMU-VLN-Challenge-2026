"""Stuck recovery: reverse a little, then retry the original goal.

The autonomy waypoint converter often will not accept a far goal when the
robot is wedged. Pulling the goal *closer* along the path also fails — if
that intermediate is inside the reach radius the stack thinks it arrived
without moving. A short backup waypoint (behind the robot) then republishing
the target is what actually frees it.
"""

from __future__ import annotations

import math


def backup_waypoint(
  robot_x: float,
  robot_y: float,
  robot_yaw: float,
  goal_x: float,
  goal_y: float,
  *,
  backup_m: float = 1.2,
  recovery_index: int = 0,
  side_offset_m: float = 0.4,
) -> tuple[float, float, float]:
  """Return a Pose2D (x, y, theta) behind the robot, away from ``goal``.

  Later recoveries add a small left/right offset so we do not reverse into
  the same hole. Heading faces the backup point so the robot drives to it.
  """
  dist_back = max(float(backup_m), 0.6)
  dx = float(goal_x) - float(robot_x)
  dy = float(goal_y) - float(robot_y)
  ahead = math.hypot(dx, dy)
  if ahead > 0.15:
    ux, uy = dx / ahead, dy / ahead
  else:
    ux = math.cos(float(robot_yaw))
    uy = math.sin(float(robot_yaw))

  bx = float(robot_x) - dist_back * ux
  by = float(robot_y) - dist_back * uy

  if int(recovery_index) > 0:
    lx, ly = -uy, ux
    side = 1.0 if int(recovery_index) % 2 else -1.0
    bx += side * float(side_offset_m) * lx
    by += side * float(side_offset_m) * ly

  theta = math.atan2(by - float(robot_y), bx - float(robot_x))
  return bx, by, theta
