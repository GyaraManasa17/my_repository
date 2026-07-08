# autocert_modules/workspace_analyzer.py

import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import alphashape
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class WorkspaceAnalyzer:

    def __init__(self, node=None):
        self.node = node

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    def load_dataset(self, dataset_file):

        print("\nLoading dataset...")
        df = pd.read_csv(dataset_file)

        total_samples = len(df)
        success_samples = df["fk_success"].sum()

        success_rate = success_samples / total_samples

        print(f"Total samples: {total_samples}")
        print(f"Successful FK samples: {success_samples}")
        print(f"FK success rate: {success_rate:.4f}")

        return df, success_rate

    # ------------------------------------------------------------

    def extract_reachable(self, df):

        print("\nExtracting reachable poses...")

        reachable = df[df["fk_success"] == 1].copy()
        xyz = reachable[["x", "y", "z"]].values

        print(f"Reachable poses: {len(reachable)}")

        return reachable, xyz

    # ------------------------------------------------------------

    def compute_xyz_limits(self, xyz):

        x_min, y_min, z_min = xyz.min(axis=0)
        x_max, y_max, z_max = xyz.max(axis=0)

        return (x_min, x_max), (y_min, y_max), (z_min, z_max)

    # ------------------------------------------------------------

    def compute_max_reach(self, xyz):

        distances = np.linalg.norm(xyz, axis=1)
        return distances.max()

    # ------------------------------------------------------------

    def compute_bounding_box_volume(self, xyz):

        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)

        dims = maxs - mins

        volume = dims[0] * dims[1] * dims[2]

        return volume

    # ------------------------------------------------------------

    def compute_convex_hull(self, xyz):

        print("\nComputing Convex Hull...")

        hull = ConvexHull(xyz)

        print(f"Convex Hull volume: {hull.volume:.6f}")
        print(f"Convex Hull area: {hull.area:.6f}")

        return hull.volume, hull.area

    # ------------------------------------------------------------

    def compute_alpha_shape(self, xyz, alpha=2.0):

        print("\nComputing Alpha Shape...")

        alpha_shape = alphashape.alphashape(xyz, alpha)

        try:
            volume = alpha_shape.volume
            area = alpha_shape.area

            print(f"Alpha Shape volume: {volume:.6f}")
            print(f"Alpha Shape area: {area:.6f}")

        except:
            volume = None
            area = None
            print("Alpha shape computation failed.")

        return alpha_shape, volume, area

    # ------------------------------------------------------------

    def compute_density(self, num_points, volume):

        if volume is None or volume == 0:
            return None

        return num_points / volume

    # ------------------------------------------------------------

    def save_reachable(self, reachable):

        print("\nSaving reachable poses...")

        reachable.to_csv("reachable_poses.csv", index=False)

    # ------------------------------------------------------------

    def save_metrics(self, metrics):

        print("Saving workspace metrics...")

        with open("workspace_metrics.txt", "w") as f:

            f.write("====================================\n")
            f.write("WORKSPACE ANALYSIS REPORT\n")
            f.write("====================================\n\n")

            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

    # ------------------------------------------------------------

    def plot_workspace_scatter(self, xyz):

        print("\nGenerating workspace scatter plot...")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            xyz[:,0],
            xyz[:,1],
            xyz[:,2],
            s=2
        )

        ax.set_title("Reachable Workspace Points")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.savefig("workspace_scatter.png", dpi=300)
        plt.close()

        print("Saved: workspace_scatter.png")

    #-------------------------------------------------------------------------

    def plot_convex_hull(self, xyz):

        print("Generating convex hull plot...")

        if len(xyz) < 4:
            print("Not enough points for workspace analysis.")
            return None

        hull = ConvexHull(xyz)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            xyz[:,0],
            xyz[:,1],
            xyz[:,2],
            s=1
        )

        for simplex in hull.simplices:
            tri = xyz[simplex]
            ax.plot(
                tri[:,0],
                tri[:,1],
                tri[:,2]
            )

        ax.set_title("Convex Hull Workspace")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.savefig("workspace_convex_hull.png", dpi=300)
        plt.close()

        print("Saved: workspace_convex_hull.png")

    #--------------------------------------------------------------------------

    def plot_alpha_shape(self, xyz, alpha=2.0):

        print("Generating alpha shape plot...")

        alpha_shape = alphashape.alphashape(xyz, alpha)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            xyz[:,0],
            xyz[:,1],
            xyz[:,2],
            s=1
        )

        try:

            if hasattr(alpha_shape, "vertices"):
                vertices = np.array(alpha_shape.vertices)
            # vertices = np.array(alpha_shape.vertices)

            ax.scatter(
                vertices[:,0],
                vertices[:,1],
                vertices[:,2],
                s=10
            )

        except:
            print("Alpha shape vertices not available for plotting")

        ax.set_title("Alpha Shape Workspace")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.savefig("workspace_alpha_shape.png", dpi=300)
        plt.close()

        print("Saved: workspace_alpha_shape.png")

    #-------------------------------------------------------------------

    def interactive_workspace_plot(self, xyz):

        print("\nGenerating interactive 3D workspace plot...")

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=xyz[:,0],
                    y=xyz[:,1],
                    z=xyz[:,2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        opacity=0.7
                    )
                )
            ]
        )

        fig.update_layout(
            title="Interactive Robot Workspace",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            )
        )

        fig.write_html("interactive_workspace.html")

        print("Saved: interactive_workspace.html")

    #------------------------------------------------------------------------

    def analyze(self, dataset_file):

        print("\n====================================")
        print("AUTO CERT WORKSPACE ANALYZER")
        print("====================================")

        df, success_rate = self.load_dataset(dataset_file)

        reachable, xyz = self.extract_reachable(df)

        if len(xyz) == 0:
            print("No reachable workspace points found.")
            return None

        num_reachable = len(reachable)

        # --------------------------------------------------------
        # XYZ Limits
        # --------------------------------------------------------

        x_lim, y_lim, z_lim = self.compute_xyz_limits(xyz)

        # --------------------------------------------------------
        # Max Reach
        # --------------------------------------------------------

        max_reach = self.compute_max_reach(xyz)

        # --------------------------------------------------------
        # Bounding Box
        # --------------------------------------------------------

        bbox_volume = self.compute_bounding_box_volume(xyz)

        # --------------------------------------------------------
        # Convex Hull
        # --------------------------------------------------------

        hull_volume, hull_area = self.compute_convex_hull(xyz)

        # --------------------------------------------------------
        # Alpha Shape
        # --------------------------------------------------------

        alpha_shape, alpha_volume, alpha_area = self.compute_alpha_shape(xyz)

        # --------------------------------------------------------
        # Density
        # --------------------------------------------------------

        bbox_density = self.compute_density(num_reachable, bbox_volume)
        alpha_density = self.compute_density(num_reachable, alpha_volume)

        # --------------------------------------------------------
        # Print summary
        # --------------------------------------------------------

        print("\n====================================")
        print("WORKSPACE METRICS SUMMARY")
        print("====================================")

        print(f"Samples total: {len(df)}")
        print(f"Reachable samples: {num_reachable}")
        print(f"FK success rate: {success_rate:.4f}")

        print("\nXYZ Limits")
        print(f"X range: [{x_lim[0]:.3f}, {x_lim[1]:.3f}]")
        print(f"Y range: [{y_lim[0]:.3f}, {y_lim[1]:.3f}]")
        print(f"Z range: [{z_lim[0]:.3f}, {z_lim[1]:.3f}]")

        print(f"\nMax radial reach: {max_reach:.4f} m")

        print(f"\nBounding box volume: {bbox_volume:.6f}")

        print(f"\nConvex hull volume: {hull_volume:.6f}")
        print(f"Convex hull area: {hull_area:.6f}")

        print(f"\nAlpha shape volume: {alpha_volume}")
        print(f"Alpha shape area: {alpha_area}")

        print(f"\nBounding box density: {bbox_density}")
        print(f"Alpha shape density: {alpha_density}")

        print("\n====================================")

        # --------------------------------------------------------
        # Save outputs
        # --------------------------------------------------------

        self.save_reachable(reachable)

        metrics = {

            "samples_total": len(df),
            "samples_reachable": num_reachable,
            "fk_success_rate": success_rate,

            "x_min": x_lim[0],
            "x_max": x_lim[1],

            "y_min": y_lim[0],
            "y_max": y_lim[1],

            "z_min": z_lim[0],
            "z_max": z_lim[1],

            "max_reach": max_reach,

            "bounding_box_volume": bbox_volume,

            "convex_hull_volume": hull_volume,
            "convex_hull_area": hull_area,

            "alpha_shape_volume": alpha_volume,
            "alpha_shape_area": alpha_area,

            "bbox_density": bbox_density,
            "alpha_density": alpha_density
        }

        self.save_metrics(metrics)
        # --------------------------------------------------------
        # Visualizations
        # --------------------------------------------------------

        self.plot_workspace_scatter(xyz)

        self.plot_convex_hull(xyz)

        self.plot_alpha_shape(xyz)

        self.interactive_workspace_plot(xyz)

        print("\nWorkspace analysis completed successfully.\n")

        return metrics