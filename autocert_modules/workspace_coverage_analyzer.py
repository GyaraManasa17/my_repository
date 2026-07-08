import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class WorkspaceCoverageAnalyzer:

    def __init__(self):
        pass

    # --------------------------------------------------

    def load_dataset(self, dataset_file):

        print("\nLoading dataset for coverage analysis...")

        df = pd.read_csv(dataset_file)

        reachable = df[df["fk_success"] == 1]

        xyz = reachable[["x","y","z"]].values

        print(f"Reachable samples: {len(xyz)}")

        return xyz

    # --------------------------------------------------

    def compute_workspace_bounds(self, xyz):

        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)

        return mins, maxs

    # --------------------------------------------------

    def voxelize_workspace(self, xyz, resolution=20):

        print("\nVoxelizing workspace...")

        mins, maxs = self.compute_workspace_bounds(xyz)

        grid_x = np.linspace(mins[0], maxs[0], resolution)
        grid_y = np.linspace(mins[1], maxs[1], resolution)
        grid_z = np.linspace(mins[2], maxs[2], resolution)

        occupied = set()

        for p in xyz:

            ix = np.searchsorted(grid_x, p[0])
            iy = np.searchsorted(grid_y, p[1])
            iz = np.searchsorted(grid_z, p[2])

            occupied.add((ix, iy, iz))

        total_voxels = resolution**3
        filled_voxels = len(occupied)

        coverage = filled_voxels / total_voxels

        print(f"Total voxels: {total_voxels}")
        print(f"Occupied voxels: {filled_voxels}")
        print(f"Workspace coverage: {coverage:.4f}")

        return coverage, occupied, grid_x, grid_y, grid_z

    # --------------------------------------------------

    def compute_uniformity(self, xyz):

        print("\nComputing sampling uniformity...")

        tree = KDTree(xyz)

        distances, _ = tree.query(xyz, k=2)

        nearest = distances[:,1]

        mean_dist = np.mean(nearest)
        std_dist = np.std(nearest)

        uniformity_score = std_dist / mean_dist

        print(f"Mean nearest neighbor distance: {mean_dist:.6f}")
        print(f"Uniformity score: {uniformity_score:.6f}")

        return uniformity_score

    # --------------------------------------------------

    def plot_voxel_coverage(self, occupied):

        print("\nGenerating voxel coverage plot...")

        xs = [v[0] for v in occupied]
        ys = [v[1] for v in occupied]
        zs = [v[2] for v in occupied]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(xs, ys, zs, s=5)

        ax.set_title("Workspace Voxel Coverage")

        plt.savefig("workspace_voxel_coverage.png", dpi=300)
        plt.close()

        print("Saved: workspace_voxel_coverage.png")

    # --------------------------------------------------

    def analyze(self, dataset_file):

        print("\n=================================")
        print("WORKSPACE COVERAGE ANALYSIS")
        print("=================================")

        xyz = self.load_dataset(dataset_file)

        coverage, occupied, gx, gy, gz = self.voxelize_workspace(xyz)

        uniformity = self.compute_uniformity(xyz)

        self.plot_voxel_coverage(occupied)

        print("\nCoverage Analysis Complete\n")

        return {
            "workspace_coverage": coverage,
            "uniformity_score": uniformity
        }