"""Coverage viewpoint planner from a map-frame LiDAR cloud.

A 360° camera already sees all headings at one XY, so this only chooses
spread-out standpoints in free space (not rotate-in-place headings).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViewpointPlannerConfig:
  num_viewpoints: int = 6
  include_start_pose: bool = True
  min_viewpoint_spacing_m: float = 3.0
  grid_resolution_m: float = 0.25
  free_clearance_m: float = 0.30
  wall_inset_m: float = 0.40
  z_min_m: float = 0.15
  z_max_m: float = 1.8
  occupied_count_min: int = 3
  # 0 or negative → use the full scanned cloud (needed for long rooms).
  max_plan_radius_m: float = 25.0
  candidate_stride_m: float = 0.5
  spacing_relax_m: float = 0.25
  occupancy_bonus: float = 0.3
  min_spacing_floor_m: float = 1.5
  failed_avoid_m: float = 1.5


@dataclass(frozen=True)
class Viewpoint:
  x: float
  y: float

  def as_tuple(self) -> tuple[float, float]:
    return (float(self.x), float(self.y))


def plan_coverage_viewpoints(
  points_xyz: np.ndarray,
  robot_xy: tuple[float, float],
  config: ViewpointPlannerConfig | None = None,
  *,
  visited: Sequence[Viewpoint] | None = None,
  exclude: Sequence[Viewpoint] | None = None,
) -> list[Viewpoint]:
  """Return up to ``num_viewpoints`` XY standpoints covering the scanned room.

  ``visited`` seeds farthest-point sampling so later stops stay away from
  snaps already taken. ``exclude`` blacklists unreachable goals.
  """
  cfg = config or ViewpointPlannerConfig()
  rx, ry = float(robot_xy[0]), float(robot_xy[1])
  start = Viewpoint(rx, ry)
  seeds = list(visited) if visited else ([] if not cfg.include_start_pose else [start])

  pts = _finite_points(points_xyz)
  if pts.size == 0:
    return seeds[: cfg.num_viewpoints] or ([start] if cfg.include_start_pose else [])

  band = pts[(pts[:, 2] >= cfg.z_min_m) & (pts[:, 2] <= cfg.z_max_m)]
  if band.shape[0] == 0:
    return seeds[: cfg.num_viewpoints] or ([start] if cfg.include_start_pose else [])

  local = _points_in_plan_radius(band, (rx, ry), cfg.max_plan_radius_m)
  if local.shape[0] == 0:
    return seeds[: cfg.num_viewpoints] or ([start] if cfg.include_start_pose else [])

  grid = _build_free_grid(local, (rx, ry), cfg)
  if grid is None:
    return seeds[: cfg.num_viewpoints] or ([start] if cfg.include_start_pose else [])

  chosen = _farthest_point_sample(
    grid, (rx, ry), cfg, seeds=seeds, exclude=exclude
  )
  if cfg.include_start_pose and not seeds:
    if not chosen or _hypot(chosen[0], start) > 0.15:
      chosen = [start] + [v for v in chosen if _hypot(v, start) >= cfg.min_spacing_floor_m]
    else:
      chosen[0] = start
  return chosen[: cfg.num_viewpoints]


def plan_next_coverage_viewpoint(
  points_xyz: np.ndarray,
  robot_xy: tuple[float, float],
  visited: Sequence[Viewpoint],
  config: ViewpointPlannerConfig | None = None,
  *,
  exclude: Sequence[Viewpoint] | None = None,
) -> Viewpoint | None:
  """Single next stop: free cell farthest from every already-visited snap."""
  cfg = config or ViewpointPlannerConfig()
  seeds = list(visited) if visited else [Viewpoint(float(robot_xy[0]), float(robot_xy[1]))]
  one = ViewpointPlannerConfig(
    num_viewpoints=len(seeds) + 1,
    include_start_pose=False,
    min_viewpoint_spacing_m=cfg.min_viewpoint_spacing_m,
    grid_resolution_m=cfg.grid_resolution_m,
    free_clearance_m=cfg.free_clearance_m,
    wall_inset_m=cfg.wall_inset_m,
    z_min_m=cfg.z_min_m,
    z_max_m=cfg.z_max_m,
    occupied_count_min=cfg.occupied_count_min,
    max_plan_radius_m=cfg.max_plan_radius_m,
    candidate_stride_m=cfg.candidate_stride_m,
    spacing_relax_m=cfg.spacing_relax_m,
    occupancy_bonus=cfg.occupancy_bonus,
    min_spacing_floor_m=cfg.min_spacing_floor_m,
    failed_avoid_m=cfg.failed_avoid_m,
  )
  planned = plan_coverage_viewpoints(
    points_xyz, robot_xy, one, visited=seeds, exclude=exclude
  )
  known = seeds + list(exclude or [])
  for view in planned:
    if all(_hypot(view, seen) >= cfg.min_spacing_floor_m for seen in known):
      return view
  return None


def suggested_viewpoint_count(
  points_xyz: np.ndarray,
  *,
  spacing_m: float = 3.0,
  min_n: int = 5,
  max_n: int = 8,
) -> int:
  """Budget snaps from the scanned long axis so a hallway gets more stops."""
  pts = _finite_points(points_xyz if points_xyz is not None else [])
  if pts.size == 0:
    return min_n
  dx = float(pts[:, 0].max() - pts[:, 0].min())
  dy = float(pts[:, 1].max() - pts[:, 1].min())
  long_axis = max(dx, dy)
  n = int(np.ceil(long_axis / max(float(spacing_m), 1.0))) + 1
  return int(max(min_n, min(max_n, n)))


def order_nearest_neighbor(
  viewpoints: list[Viewpoint],
  start_xy: tuple[float, float],
) -> list[Viewpoint]:
  """Greedy tour starting at ``start_xy`` (or the closest listed viewpoint)."""
  if not viewpoints:
    return []

  remaining = list(viewpoints)
  sx, sy = float(start_xy[0]), float(start_xy[1])
  start_idx = min(
    range(len(remaining)),
    key=lambda i: _hypot(remaining[i], Viewpoint(sx, sy)),
  )
  ordered = [remaining.pop(start_idx)]
  while remaining:
    cur = ordered[-1]
    nxt = min(range(len(remaining)), key=lambda i: _hypot(remaining[i], cur))
    ordered.append(remaining.pop(nxt))
  return ordered


def _points_in_plan_radius(
  band: np.ndarray,
  robot_xy: tuple[float, float],
  max_radius_m: float,
) -> np.ndarray:
  if max_radius_m <= 0:
    return band
  dist = np.hypot(band[:, 0] - robot_xy[0], band[:, 1] - robot_xy[1])
  return band[dist <= max_radius_m]


def _finite_points(points_xyz: np.ndarray) -> np.ndarray:
  if points_xyz is None or len(points_xyz) == 0:
    return np.empty((0, 3), dtype=np.float64)
  pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
  return pts[np.isfinite(pts).all(axis=1)]


def _hypot(a: Viewpoint, b: Viewpoint) -> float:
  return float(math_hypot(a.x - b.x, a.y - b.y))


def math_hypot(dx: float, dy: float) -> float:
  return float(np.hypot(dx, dy))


@dataclass
class _Grid:
  origin_x: float
  origin_y: float
  resolution: float
  occupied: np.ndarray
  inflated: np.ndarray
  free: np.ndarray
  dist_to_occupied_m: np.ndarray

  def cell_center(self, ix: int, iy: int) -> Viewpoint:
    return Viewpoint(
      self.origin_x + (ix + 0.5) * self.resolution,
      self.origin_y + (iy + 0.5) * self.resolution,
    )

  def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
    ix = int(np.floor((x - self.origin_x) / self.resolution))
    iy = int(np.floor((y - self.origin_y) / self.resolution))
    return ix, iy


def _build_free_grid(
  local: np.ndarray,
  robot_xy: tuple[float, float],
  cfg: ViewpointPlannerConfig,
) -> _Grid | None:
  res = float(cfg.grid_resolution_m)
  if res <= 0:
    return None

  rx, ry = robot_xy
  pad = float(cfg.free_clearance_m) + float(cfg.wall_inset_m) + res
  xmin = float(min(local[:, 0].min(), rx) - pad)
  xmax = float(max(local[:, 0].max(), rx) + pad)
  ymin = float(min(local[:, 1].min(), ry) - pad)
  ymax = float(max(local[:, 1].max(), ry) + pad)

  width = max(int(np.ceil((xmax - xmin) / res)), 1)
  height = max(int(np.ceil((ymax - ymin) / res)), 1)
  # Guard huge / empty degenerates.
  if width * height > 250_000:
    return None

  counts = np.zeros((height, width), dtype=np.int32)
  ix = np.floor((local[:, 0] - xmin) / res).astype(np.int32)
  iy = np.floor((local[:, 1] - ymin) / res).astype(np.int32)
  valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
  ix, iy = ix[valid], iy[valid]
  np.add.at(counts, (iy, ix), 1)

  occupied = counts >= int(cfg.occupied_count_min)
  inflate_r = int(np.ceil(cfg.free_clearance_m / res))
  inflated = _dilate_circle(occupied, inflate_r)

  hull = _convex_hull_xy(local[:, :2])
  free = np.zeros((height, width), dtype=bool)
  xs = xmin + (np.arange(width) + 0.5) * res
  ys = ymin + (np.arange(height) + 0.5) * res
  xx, yy = np.meshgrid(xs, ys)
  if len(hull) >= 3:
    inside = _points_in_convex_hull(xx.ravel(), yy.ravel(), hull).reshape(height, width)
  else:
    inside = (
      (xx >= local[:, 0].min())
      & (xx <= local[:, 0].max())
      & (yy >= local[:, 1].min())
      & (yy <= local[:, 1].max())
    )
  if cfg.max_plan_radius_m <= 0:
    in_radius = np.ones_like(xx, dtype=bool)
  else:
    in_radius = np.hypot(xx - rx, yy - ry) <= cfg.max_plan_radius_m
  free = inside & in_radius & ~inflated

  rix, riy = int(np.floor((rx - xmin) / res)), int(np.floor((ry - ymin) / res))
  if 0 <= rix < width and 0 <= riy < height:
    free[riy, rix] = True

  dist_occ = _distance_to_true(occupied, res)
  return _Grid(
    origin_x=xmin,
    origin_y=ymin,
    resolution=res,
    occupied=occupied,
    inflated=inflated,
    free=free,
    dist_to_occupied_m=dist_occ,
  )


def _farthest_point_sample(
  grid: _Grid,
  robot_xy: tuple[float, float],
  cfg: ViewpointPlannerConfig,
  *,
  seeds: Sequence[Viewpoint] | None = None,
  exclude: Sequence[Viewpoint] | None = None,
) -> list[Viewpoint]:
  stride = max(1, int(round(cfg.candidate_stride_m / grid.resolution)))
  all_candidates = _candidate_cells(grid, stride=stride, min_occ_dist=0.0)
  inset_candidates = _candidate_cells(grid, stride=stride, min_occ_dist=cfg.wall_inset_m)
  if not all_candidates:
    return [Viewpoint(robot_xy[0], robot_xy[1])] if cfg.include_start_pose else []

  chosen: list[Viewpoint] = list(seeds) if seeds else []
  if not chosen and cfg.include_start_pose:
    chosen.append(Viewpoint(robot_xy[0], robot_xy[1]))

  spacing = float(cfg.min_viewpoint_spacing_m)
  pool = inset_candidates or all_candidates
  while len(chosen) < int(cfg.num_viewpoints):
    picked = _best_candidate(
      pool,
      chosen,
      grid,
      spacing,
      cfg.occupancy_bonus,
      exclude=exclude,
      exclude_radius=cfg.failed_avoid_m,
    )
    if picked is None:
      if pool is inset_candidates and inset_candidates is not all_candidates:
        pool = all_candidates
        continue
      spacing -= float(cfg.spacing_relax_m)
      if spacing < float(cfg.min_spacing_floor_m):
        break
      continue
    chosen.append(picked)

  return chosen


def _candidate_cells(
  grid: _Grid,
  *,
  stride: int,
  min_occ_dist: float,
) -> list[tuple[int, int]]:
  height, width = grid.free.shape
  cells: list[tuple[int, int]] = []
  for iy in range(0, height, stride):
    for ix in range(0, width, stride):
      if not grid.free[iy, ix]:
        continue
      if grid.dist_to_occupied_m[iy, ix] < min_occ_dist:
        continue
      cells.append((ix, iy))
  return cells


def _best_candidate(
  cells: list[tuple[int, int]],
  chosen: list[Viewpoint],
  grid: _Grid,
  spacing: float,
  occupancy_bonus: float,
  *,
  exclude: Sequence[Viewpoint] | None = None,
  exclude_radius: float = 1.5,
) -> Viewpoint | None:
  best: Viewpoint | None = None
  best_score = -1.0
  blocked = list(exclude or [])
  for ix, iy in cells:
    view = grid.cell_center(ix, iy)
    if blocked and min(_hypot(view, b) for b in blocked) < exclude_radius:
      continue
    if chosen:
      mind = min(_hypot(view, c) for c in chosen)
      if mind < spacing:
        continue
    else:
      mind = 0.0
    score = mind + occupancy_bonus * float(grid.dist_to_occupied_m[iy, ix])
    if score > best_score:
      best_score = score
      best = view
  return best


def _dilate_circle(mask: np.ndarray, radius_cells: int) -> np.ndarray:
  if radius_cells <= 0 or not mask.any():
    return mask.copy()
  ys, xs = np.where(mask)
  out = mask.copy()
  height, width = mask.shape
  for dy in range(-radius_cells, radius_cells + 1):
    for dx in range(-radius_cells, radius_cells + 1):
      if dx * dx + dy * dy > radius_cells * radius_cells:
        continue
      yy = ys + dy
      xx = xs + dx
      valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
      out[yy[valid], xx[valid]] = True
  return out


def _distance_to_true(mask: np.ndarray, resolution: float) -> np.ndarray:
  height, width = mask.shape
  dist = np.full((height, width), 1e6, dtype=np.float64)
  if not mask.any():
    return np.zeros((height, width), dtype=np.float64)
  occ_y, occ_x = np.where(mask)
  yy = np.arange(height)[:, None]
  xx = np.arange(width)[None, :]
  # Chunk occupied cells to keep memory bounded on large rooms.
  chunk = 256
  for start in range(0, occ_x.size, chunk):
    ox = occ_x[start : start + chunk][None, None, :]
    oy = occ_y[start : start + chunk][None, None, :]
    d = np.hypot(
      (xx[:, :, None] - ox) * resolution,
      (yy[:, :, None] - oy) * resolution,
    )
    dist = np.minimum(dist, d.min(axis=2))
  return dist


def _convex_hull_xy(xy: np.ndarray) -> list[tuple[float, float]]:
  pts = sorted({(float(x), float(y)) for x, y in xy})
  if len(pts) <= 2:
    return pts

  def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

  lower: list[tuple[float, float]] = []
  for p in pts:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
      lower.pop()
    lower.append(p)
  upper: list[tuple[float, float]] = []
  for p in reversed(pts):
    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
      upper.pop()
    upper.append(p)
  return lower[:-1] + upper[:-1]


def _points_in_convex_hull(
  xs: np.ndarray,
  ys: np.ndarray,
  hull: list[tuple[float, float]],
) -> np.ndarray:
  n = len(hull)
  inside = np.ones(xs.shape, dtype=bool)
  sign = np.zeros(xs.shape, dtype=np.int8)
  for i in range(n):
    x1, y1 = hull[i]
    x2, y2 = hull[(i + 1) % n]
    cross = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
    nonzero = np.abs(cross) > 1e-9
    s = np.where(cross > 0, 1, -1).astype(np.int8)
    unset = nonzero & (sign == 0)
    sign[unset] = s[unset]
    inside &= ~nonzero | (sign == s)
  return inside
