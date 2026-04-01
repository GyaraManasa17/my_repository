#!/usr/bin/env python3

import pandas as pd
import numpy as np
import glob
import os
import argparse
from scipy.spatial import Delaunay
import math


def generate_benchmark_poses(total_poses=100, interior_ratio=0.8):

    num_interior = int(total_poses * interior_ratio)
    num_extremes = total_poses - num_interior

    # 1. Auto-find newest boundary CSV
    files = glob.glob("omx_workspace_boundary_*.csv")
    if not files:
        print("❌ No boundary CSV files found! Run workspace analyzer first.")
        return
    csv_filename = max(files, key=os.path.getctime)
    print(f"📂 Loading workspace boundaries from: {csv_filename}")

    # 2. Load boundary
    df = pd.read_csv(csv_filename)
    boundary_points = df[['x', 'y', 'z']].values

    min_x, max_x = boundary_points[:, 0].min(), boundary_points[:, 0].max()
    min_y, max_y = boundary_points[:, 1].min(), boundary_points[:, 1].max()
    min_z, max_z = boundary_points[:, 2].min(), boundary_points[:, 2].max()

    print("⚙️ Building 3D hull mesh...")
    hull_mesh = Delaunay(boundary_points)

    # 3. Interior sampling
    print(f"🎲 Generating {num_interior} interior poses...")
    interior_poses = []
    while len(interior_poses) < num_interior:
        pt = [
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y),
            np.random.uniform(min_z, max_z)
        ]

        if hull_mesh.find_simplex(pt) >= 0:
            interior_poses.append({
                'type': 'interior',
                'x': pt[0], 'y': pt[1], 'z': pt[2]
            })

    # 4. Extreme sampling
    print(f"⚡ Selecting {num_extremes} boundary poses...")
    idx_choices = np.random.choice(len(boundary_points), num_extremes, replace=False)

    extreme_poses = []
    for idx in idx_choices:
        pt = boundary_points[idx]
        extreme_poses.append({
            'type': 'extreme_edge',
            'x': pt[0], 'y': pt[1], 'z': pt[2]
        })

    all_poses = interior_poses + extreme_poses

    # 5. Add orientation
    for pose in all_poses:
        pose['roll'] = np.random.uniform(-math.pi, math.pi)
        pose['pitch'] = np.random.uniform(-math.pi/2, math.pi/2)
        pose['yaw'] = np.random.uniform(-math.pi, math.pi)

    # 6. Save
    out_df = pd.DataFrame(all_poses)
    out_filename = f"robot_efficiency_benchmark_{total_poses}.csv"
    out_df.to_csv(out_filename, index=False)

    print("\n" + "="*60)
    print(f"✅ Generated {len(out_df)} Benchmark Poses")
    print(f"💾 Saved to: {out_filename}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate robot benchmarking poses")
    parser.add_argument("--total", type=int, default=100,
                        help="Total number of poses to generate")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="Interior pose ratio (0.0 - 1.0)")

    args = parser.parse_args()

    generate_benchmark_poses(
        total_poses=args.total,
        interior_ratio=args.ratio
    )