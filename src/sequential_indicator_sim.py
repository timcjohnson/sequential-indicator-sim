#!/usr/bin/env python3
"""Sequential indicator simulation on a 3D grid with a Gaussian variogram."""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


def gaussian_variogram(h2: np.ndarray, nugget: float) -> np.ndarray:
    """Return Gaussian variogram gamma(h) with precomputed scaled distance^2."""
    h2 = np.asarray(h2, dtype=float)
    sill = 1.0
    if nugget < 0.0 or nugget > sill:
        raise ValueError("nugget must be in [0, 1]")
    return nugget + (sill - nugget) * (1.0 - np.exp(-h2))


def gaussian_covariance(h2: np.ndarray, nugget: float) -> np.ndarray:
    """Return covariance from Gaussian variogram using scaled distance^2."""
    sill = 1.0
    gamma = gaussian_variogram(h2, nugget)
    return sill - gamma


def scaled_distance_sq(diffs: np.ndarray, vrange_xyz: Sequence[float]) -> np.ndarray:
    """Compute scaled squared distance using per-axis ranges."""
    vr = np.asarray(vrange_xyz, dtype=float)
    if np.any(vr <= 0.0):
        raise ValueError("vrange values must be > 0")
    scaled = diffs / vr
    return np.sum(scaled ** 2, axis=-1)


@dataclass
class GridSpec:
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    num: Tuple[int, int, int]

    def generate_points(self) -> np.ndarray:
        nx, ny, nz = self.num
        sx, sy, sz = self.start
        ex, ey, ez = self.end
        xs = np.linspace(sx, ex, nx)
        ys = np.linspace(sy, ey, ny)
        zs = np.linspace(sz, ez, nz)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def load_known_points(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load known points from whitespace-delimited text: x y z category."""
    xyz: List[Tuple[float, float, float]] = []
    cats: List[int] = []
    with open(path, "r", newline="") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError("Known points must have 4 columns: x y z category")
            x, y, z = (float(parts[0]), float(parts[1]), float(parts[2]))
            cat = int(parts[3])
            xyz.append((x, y, z))
            cats.append(cat)
    return np.asarray(xyz, dtype=float), np.asarray(cats, dtype=int)


def build_known_index(xyz: np.ndarray, cats: np.ndarray, tol: float) -> Dict[Tuple[int, int, int], int]:
    """Index known points by quantized coordinates for fast lookup."""
    if xyz.shape[0] == 0:
        return {}
    scale = 1.0 / tol
    keys = np.round(xyz * scale).astype(int)
    return {tuple(key): int(cat) for key, cat in zip(keys, cats)}


def nearest_neighbors(
    points: np.ndarray,
    target: np.ndarray,
    max_neighbors: int,
    search_radius_xyz: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return neighbor indices and distances using anisotropic search radii."""
    if points.shape[0] == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=float)
    diffs = points - target[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    
    # Apply anisotropic search ellipsoid
    search_radii = np.asarray(search_radius_xyz, dtype=float)
    if np.all(np.isfinite(search_radii)) and np.all(search_radii > 0):
        scaled_diffs = diffs / search_radii
        scaled_dists = np.linalg.norm(scaled_diffs, axis=1)
        mask = scaled_dists <= 1.0
    else:
        mask = np.ones_like(dists, dtype=bool)
    
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=float)
    if idx.size > max_neighbors:
        sub = np.argpartition(dists[idx], max_neighbors)[:max_neighbors]
        idx = idx[sub]
    return idx, dists[idx]


def ordinary_kriging_weights(
    neighbor_xyz: np.ndarray,
    target: np.ndarray,
    vrange_xyz: Sequence[float],
    nugget: float,
) -> np.ndarray:
    """Compute ordinary kriging weights for neighbors."""
    n = neighbor_xyz.shape[0]
    if n == 0:
        return np.empty((0,), dtype=float)
    diffs = neighbor_xyz[:, None, :] - neighbor_xyz[None, :, :]
    dists_sq = scaled_distance_sq(diffs, vrange_xyz)
    cov = gaussian_covariance(dists_sq, nugget)
    d0_sq = scaled_distance_sq(neighbor_xyz - target[None, :], vrange_xyz)
    cov0 = gaussian_covariance(d0_sq, nugget)

    mat = np.zeros((n + 1, n + 1), dtype=float)
    mat[:n, :n] = cov
    mat[:n, n] = 1.0
    mat[n, :n] = 1.0
    rhs = np.zeros((n + 1,), dtype=float)
    rhs[:n] = cov0
    rhs[n] = 1.0

    try:
        sol = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        sol, _, _, _ = np.linalg.lstsq(mat, rhs, rcond=None)
    return sol[:n]


def compute_probs_for_categories(
    categories: Sequence[int],
    neighbor_cats: np.ndarray,
    weights: np.ndarray,
    fallback_props: np.ndarray,
) -> np.ndarray:
    """Compute probabilities for each category from indicator kriging."""
    probs = np.zeros((len(categories),), dtype=float)
    for i, cat in enumerate(categories):
        indicator = (neighbor_cats == cat).astype(float)
        if weights.size == 0:
            probs[i] = fallback_props[i]
        else:
            probs[i] = float(np.dot(weights, indicator))
    probs = np.clip(probs, 0.0, 1.0)
    total = float(probs.sum())
    if total <= 0.0:
        return fallback_props
    return probs / total


def proportions(categories: Sequence[int], cats: np.ndarray) -> np.ndarray:
    """Compute category proportions for fallback."""
    counts = np.array([np.sum(cats == cat) for cat in categories], dtype=float)
    total = float(counts.sum())
    if total <= 0.0:
        return np.full((len(categories),), 1.0 / len(categories), dtype=float)
    return counts / total


def sequential_indicator_simulation(
    grid: np.ndarray,
    known_xyz: np.ndarray,
    known_cats: np.ndarray,
    categories: Sequence[int],
    vrange_xyz: Sequence[float],
    nugget: float,
    max_neighbors: int,
    search_radius_xyz: Sequence[float],
    seed: int | None,
    tol: float,
) -> np.ndarray:
    """Run sequential indicator simulation over the grid."""
    rng = np.random.default_rng(seed)
    n_nodes = grid.shape[0]
    sim_cats = np.full((n_nodes,), -1, dtype=int)

    known_index = build_known_index(known_xyz, known_cats, tol)
    scale = 1.0 / tol if tol > 0 else 1.0

    # Conditioning data updated as we simulate
    cond_xyz = np.array(known_xyz, dtype=float, copy=True)
    cond_cats = np.array(known_cats, dtype=int, copy=True)
    base_props = proportions(categories, known_cats)
    min_neighbors = min(4, max_neighbors)

    order = rng.permutation(n_nodes)
    for idx in order:
        node = grid[idx]
        key = tuple(np.round(node * scale).astype(int))
        if key in known_index:
            sim_cats[idx] = known_index[key]
            continue

        neighbor_idx, _ = nearest_neighbors(cond_xyz, node, max_neighbors, search_radius_xyz)
        if neighbor_idx.size < min_neighbors:
            probs = base_props
        else:
            neighbor_xyz = cond_xyz[neighbor_idx]
            neighbor_cats = cond_cats[neighbor_idx]
            weights = ordinary_kriging_weights(neighbor_xyz, node, vrange_xyz, nugget)
            probs = compute_probs_for_categories(categories, neighbor_cats, weights, base_props)

        sim_cat = int(rng.choice(categories, p=probs))
        sim_cats[idx] = sim_cat

        cond_xyz = np.vstack([cond_xyz, node[None, :]])
        cond_cats = np.append(cond_cats, sim_cat)

    return sim_cats


def compute_interpolation_indices(
    coarse_grid: np.ndarray,
    fine_grid: np.ndarray,
) -> np.ndarray:
    """Precompute nearest-neighbor indices for interpolation using KDTree."""
    tree = cKDTree(coarse_grid)
    _, nearest_indices = tree.query(fine_grid, k=1)
    return nearest_indices


def apply_interpolation(
    coarse_cats: np.ndarray,
    interp_indices: np.ndarray,
) -> np.ndarray:
    """Apply precomputed interpolation indices to category data."""
    return coarse_cats[interp_indices]


def write_output_npz(path: str, grid: np.ndarray, sim_cats: np.ndarray) -> None:
    with open(path.replace('.npz', ''), 'wb') as f:
        np.savez_compressed(
            f,
            x=grid[:, 0],
            y=grid[:, 1],
            z=grid[:, 2],
            category=sim_cats
        )


def _run_single_realization(
    args_tuple: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, Sequence[int], 
                       Sequence[float], float, int, Sequence[float], float, np.ndarray, np.ndarray],
) -> Tuple[int, np.ndarray]:
    """Worker function to run a single realization. Returns (realization_num, output_cats)."""
    (realization, seed, grid, known_xyz, known_cats, categories, vrange_xyz, nugget,
     max_neighbors, search_radius_xyz, tol, interp_indices, base_known_index) = args_tuple
    
    # Set seed for this realization
    real_seed = seed + realization if seed is not None else None
    
    # Run simulation
    sim_cats = sequential_indicator_simulation(
        grid=grid,
        known_xyz=known_xyz,
        known_cats=known_cats,
        categories=categories,
        vrange_xyz=vrange_xyz,
        nugget=nugget,
        max_neighbors=max_neighbors,
        search_radius_xyz=search_radius_xyz,
        seed=real_seed,
        tol=tol,
    )
    
    # Apply interpolation if needed
    if interp_indices is not None:
        output_cats = apply_interpolation(sim_cats, interp_indices)
    else:
        output_cats = sim_cats
    
    return (realization, output_cats)


def parse_categories(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("categories list is empty")
    return [int(p) for p in parts]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequential indicator simulation with Gaussian variogram on a 3D grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--known", help="TXT with whitespace-delimited x y z category")
    parser.add_argument("--output", required=True, help="Output NPZ path (without extension)")
    parser.add_argument("--x-start", type=float, required=True)
    parser.add_argument("--x-end", type=float, required=True)
    parser.add_argument("--x-num", type=int, required=True)
    parser.add_argument("--y-start", type=float, required=True)
    parser.add_argument("--y-end", type=float, required=True)
    parser.add_argument("--y-num", type=int, required=True)
    parser.add_argument("--z-start", type=float, required=True)
    parser.add_argument("--z-end", type=float, required=True)
    parser.add_argument("--z-num", type=int, required=True)
    parser.add_argument("--categories", type=str, default="")
    parser.add_argument("--vrange-x", type=float, required=True, help="Gaussian range in x")
    parser.add_argument("--vrange-y", type=float, required=True, help="Gaussian range in y")
    parser.add_argument("--vrange-z", type=float, required=True, help="Gaussian range in z")
    parser.add_argument("--nugget", type=float, default=0.0)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--search-radius-x", type=float, default=float("inf"), help="Search radius in x")
    parser.add_argument("--search-radius-y", type=float, default=float("inf"), help="Search radius in y")
    parser.add_argument("--search-radius-z", type=float, default=float("inf"), help="Search radius in z")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for matching known points")
    parser.add_argument("--interp-x-num", type=int, help="Interpolate to finer grid with this many x points")
    parser.add_argument("--interp-y-num", type=int, help="Interpolate to finer grid with this many y points")
    parser.add_argument("--interp-z-num", type=int, help="Interpolate to finer grid with this many z points")
    parser.add_argument("--num-realizations", type=int, default=1, help="Number of simulation realizations to generate")
    parser.add_argument("--num-cores", type=int, help="Number of CPU cores to use for parallelization (default: all available)")

    args = parser.parse_args()

    if args.known:
        known_xyz, known_cats = load_known_points(args.known)
    else:
        known_xyz = np.empty((0, 3), dtype=float)
        known_cats = np.empty((0,), dtype=int)

    if args.categories:
        categories = parse_categories(args.categories)
    else:
        categories = sorted(int(c) for c in np.unique(known_cats))
    if not categories:
        raise ValueError("No categories provided. Use --categories or provide --known data.")

    grid_spec = GridSpec(
        start=(args.x_start, args.y_start, args.z_start),
        end=(args.x_end, args.y_end, args.z_end),
        num=(args.x_num, args.y_num, args.z_num),
    )
    grid = grid_spec.generate_points()

    # Precompute interpolation indices if needed
    interp_indices = None
    output_grid = grid
    if args.interp_x_num or args.interp_y_num or args.interp_z_num:
        fine_nx = args.interp_x_num if args.interp_x_num else args.x_num
        fine_ny = args.interp_y_num if args.interp_y_num else args.y_num
        fine_nz = args.interp_z_num if args.interp_z_num else args.z_num
        
        fine_grid_spec = GridSpec(
            start=(args.x_start, args.y_start, args.z_start),
            end=(args.x_end, args.y_end, args.z_end),
            num=(fine_nx, fine_ny, fine_nz),
        )
        fine_grid = fine_grid_spec.generate_points()
        interp_indices = compute_interpolation_indices(grid, fine_grid)
        output_grid = fine_grid

    # Generate multiple realizations in parallel
    if args.num_realizations > 1 and args.num_cores != 1:
        num_workers = args.num_cores if args.num_cores else None
        
        # Prepare arguments for worker function
        worker_args = [
            (
                realization,
                args.seed,
                grid,
                known_xyz,
                known_cats,
                categories,
                (args.vrange_x, args.vrange_y, args.vrange_z),
                args.nugget,
                args.max_neighbors,
                (args.search_radius_x, args.search_radius_y, args.search_radius_z),
                args.tol,
                interp_indices,
                None,
            )
            for realization in range(args.num_realizations)
        ]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_run_single_realization, worker_args)
        
        # Write results in order
        for realization, output_cats in results:
            if args.num_realizations > 1:
                output_path = f"{args.output}_{realization + 1}.npz"
            else:
                output_path = f"{args.output}.npz"
            
            write_output_npz(output_path, output_grid, output_cats)
    else:
        # Serial execution
        for realization in range(args.num_realizations):
            # Set seed for this realization if base seed provided
            real_seed = args.seed + realization if args.seed is not None else None
            
            sim_cats = sequential_indicator_simulation(
                grid=grid,
                known_xyz=known_xyz,
                known_cats=known_cats,
                categories=categories,
                vrange_xyz=(args.vrange_x, args.vrange_y, args.vrange_z),
                nugget=args.nugget,
                max_neighbors=args.max_neighbors,
                search_radius_xyz=(args.search_radius_x, args.search_radius_y, args.search_radius_z),
                seed=real_seed,
                tol=args.tol,
            )

            # Apply interpolation if needed
            if interp_indices is not None:
                output_cats = apply_interpolation(sim_cats, interp_indices)
            else:
                output_cats = sim_cats

            # Generate output filename
            if args.num_realizations > 1:
                output_path = f"{args.output}_{realization + 1}.npz"
            else:
                output_path = f"{args.output}.npz"
            
            write_output_npz(output_path, output_grid, output_cats)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
