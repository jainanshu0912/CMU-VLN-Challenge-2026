"""Unit tests for LiDAR coverage viewpoint planning."""

from __future__ import annotations

import unittest

import numpy as np

from vlm_pipeline_live.viewpoint_planner import (
  Viewpoint,
  ViewpointPlannerConfig,
  order_nearest_neighbor,
  plan_coverage_viewpoints,
  plan_next_coverage_viewpoint,
  suggested_viewpoint_count,
)


def _wall_rectangle(
  xmin: float,
  xmax: float,
  ymin: float,
  ymax: float,
  *,
  z: float = 0.8,
  step: float = 0.1,
) -> np.ndarray:
  xs = np.arange(xmin, xmax + 1e-6, step)
  ys = np.arange(ymin, ymax + 1e-6, step)
  pts = []
  for x in xs:
    pts.append([x, ymin, z])
    pts.append([x, ymax, z])
  for y in ys:
    pts.append([xmin, y, z])
    pts.append([xmax, y, z])
  return np.asarray(pts, dtype=np.float64)


def _filled_box(
  xmin: float,
  xmax: float,
  ymin: float,
  ymax: float,
  *,
  z: float = 0.8,
  step: float = 0.1,
) -> np.ndarray:
  xs = np.arange(xmin, xmax + 1e-6, step)
  ys = np.arange(ymin, ymax + 1e-6, step)
  xx, yy = np.meshgrid(xs, ys)
  zz = np.full(xx.shape, z)
  return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


class ViewpointPlannerTest(unittest.TestCase):
  def test_empty_cloud_returns_start(self) -> None:
    views = plan_coverage_viewpoints(np.empty((0, 3)), (1.5, -0.2))
    self.assertEqual(len(views), 1)
    self.assertAlmostEqual(views[0].x, 1.5, places=5)
    self.assertAlmostEqual(views[0].y, -0.2, places=5)

  def test_spreads_views_in_open_room(self) -> None:
    walls = _wall_rectangle(-4.0, 4.0, -3.0, 3.0)
    cfg = ViewpointPlannerConfig(num_viewpoints=6, min_viewpoint_spacing_m=2.5)
    views = plan_coverage_viewpoints(walls, (0.0, 0.0), cfg)
    self.assertGreaterEqual(len(views), 4)
    self.assertLessEqual(len(views), 6)
    self.assertAlmostEqual(views[0].x, 0.0, places=1)
    self.assertAlmostEqual(views[0].y, 0.0, places=1)
    for view in views:
      self.assertGreater(view.x, -3.6)
      self.assertLess(view.x, 3.6)
      self.assertGreater(view.y, -2.6)
      self.assertLess(view.y, 2.6)
    others = views[1:]
    if others:
      nearest = min(np.hypot(v.x, v.y) for v in others)
      self.assertGreaterEqual(nearest, 1.0)

  def test_avoids_occupied_furniture(self) -> None:
    walls = _wall_rectangle(-4.0, 4.0, -3.0, 3.0)
    sofa = _filled_box(1.2, 2.4, -0.6, 0.6)
    cloud = np.vstack([walls, sofa])
    cfg = ViewpointPlannerConfig(num_viewpoints=6, min_viewpoint_spacing_m=2.0)
    views = plan_coverage_viewpoints(cloud, (0.0, 0.0), cfg)
    for view in views[1:]:
      inside_sofa = (1.0 <= view.x <= 2.6) and (-0.8 <= view.y <= 0.8)
      self.assertFalse(inside_sofa, f"viewpoint landed in sofa {view}")

  def test_order_nearest_neighbor_starts_at_robot(self) -> None:
    views = [
      Viewpoint(4.0, 0.0),
      Viewpoint(0.0, 0.0),
      Viewpoint(4.0, 3.0),
    ]
    ordered = order_nearest_neighbor(views, (0.05, 0.0))
    self.assertAlmostEqual(ordered[0].x, 0.0)
    self.assertAlmostEqual(ordered[0].y, 0.0)
    self.assertAlmostEqual(ordered[1].x, 4.0)
    self.assertAlmostEqual(ordered[1].y, 0.0)

  def test_long_hallway_reaches_far_end(self) -> None:
    walls = _wall_rectangle(-2.5, 2.5, -12.0, 2.5)
    cfg = ViewpointPlannerConfig(
      num_viewpoints=6,
      min_viewpoint_spacing_m=3.0,
      max_plan_radius_m=25.0,
    )
    views = plan_coverage_viewpoints(walls, (0.0, 0.0), cfg)
    self.assertGreaterEqual(len(views), 4)
    south = min(v.y for v in views)
    self.assertLess(south, -8.0, f"tour stayed north; ys={[round(v.y, 2) for v in views]}")

  def test_next_stop_is_farthest_from_visited(self) -> None:
    walls = _wall_rectangle(-2.5, 2.5, -12.0, 2.5)
    cfg = ViewpointPlannerConfig(
      num_viewpoints=6,
      min_viewpoint_spacing_m=3.0,
      max_plan_radius_m=25.0,
    )
    visited = [Viewpoint(0.0, 0.0), Viewpoint(0.0, -3.5)]
    nxt = plan_next_coverage_viewpoint(walls, (0.0, -3.5), visited, cfg)
    self.assertIsNotNone(nxt)
    assert nxt is not None
    self.assertLess(nxt.y, -7.0, f"next stop did not push south: {nxt}")

  def test_failed_goal_is_not_rechosen(self) -> None:
    walls = _wall_rectangle(-2.5, 2.5, -12.0, 2.5)
    cfg = ViewpointPlannerConfig(
      num_viewpoints=6,
      min_viewpoint_spacing_m=3.0,
      max_plan_radius_m=25.0,
      failed_avoid_m=2.0,
    )
    visited = [Viewpoint(0.0, 0.0)]
    first = plan_next_coverage_viewpoint(walls, (0.0, 0.0), visited, cfg)
    self.assertIsNotNone(first)
    assert first is not None
    second = plan_next_coverage_viewpoint(
      walls, (0.0, 0.0), visited, cfg, exclude=[first]
    )
    self.assertIsNotNone(second)
    assert second is not None
    self.assertGreater(
      float(np.hypot(first.x - second.x, first.y - second.y)),
      1.5,
    )

  def test_suggested_count_grows_with_long_axis(self) -> None:
    short = _wall_rectangle(-3.0, 3.0, -3.0, 3.0)
    long = _wall_rectangle(-2.5, 2.5, -12.0, 2.5)
    self.assertGreaterEqual(suggested_viewpoint_count(long, spacing_m=3.0), 6)
    self.assertLessEqual(suggested_viewpoint_count(short, spacing_m=3.0), 6)


if __name__ == "__main__":
  unittest.main()
