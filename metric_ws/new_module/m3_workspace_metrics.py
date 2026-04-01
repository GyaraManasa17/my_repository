import os
import glob
import pandas as pd
import numpy as np

# NEW: Headless backend configuration to prevent GUI crashes in ROS 2 / Docker
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from scipy.stats import entropy
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
import alphashape
import trimesh

class WorkspaceAnalyzer:

    def __init__(self, node):
        self.node = node
        
        # Define directories
        self.output_dir = "workspace_metrics"
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.distributions_dir = os.path.join(self.output_dir, "distributions")

        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.distributions_dir, exist_ok=True)
        
        self.node.get_logger().info("📂 WorkspaceAnalyzer initialized. Directories created/verified.")

    # --------------------------------------------------------
    # AUTO LOAD DATASET FROM MODULE 2
    # --------------------------------------------------------
    def load_latest_dataset(self):
        self.node.get_logger().info("🔍 Searching for the latest workspace dataset CSV...")
        
        csv_files = glob.glob("workspace_dataset_*.csv")
        if not csv_files:
            csv_files = glob.glob("workspace_fk_dataset_*.csv")

        if not csv_files:
            self.node.get_logger().error("❌ No workspace dataset CSV found! Module 2 may have failed.")
            return None

        latest_dataset = max(csv_files, key=os.path.getctime)
        self.node.get_logger().info(f"✅ Found dataset for analysis: {latest_dataset}")
        return latest_dataset

    def compute_alpha_shape_volume(self, points, alpha=2.0):
        """
        Compute concave hull (Alpha Shape) volume.
        Returns volume and mesh. Includes smart downsampling to prevent freezing.
        """
        try:
            # NEW: Smart Boundary-Preserving Downsampling
            max_pts = 5000
            if len(points) > max_pts:
                self.node.get_logger().info(f"   ➜ Downsampling {len(points)} points to {max_pts} for Alpha Shape...")
                
                # 1. Guarantee the outer boundary is perfectly preserved!
                hull = ConvexHull(points)
                boundary_points = points[hull.vertices]
                
                # 2. Sample the remaining internal points uniformly
                remaining_indices = list(set(range(len(points))) - set(hull.vertices))
                sample_size = min(max_pts - len(boundary_points), len(remaining_indices))
                
                if sample_size > 0:
                    sampled_indices = np.random.choice(remaining_indices, sample_size, replace=False)
                    sampled_points = np.vstack((boundary_points, points[sampled_indices]))
                else:
                    sampled_points = boundary_points
            else:
                sampled_points = points

            self.node.get_logger().info("   ➜ Computing Alpha Shape (Concave Hull)...")
            alpha_shape = alphashape.alphashape(sampled_points, alpha)

            if hasattr(alpha_shape, "vertices"):
                mesh = trimesh.Trimesh(
                    vertices=np.array(alpha_shape.vertices),
                    faces=np.array(alpha_shape.faces)
                )

                volume = mesh.volume
                return float(volume), mesh

            return 0.0, None

        except Exception as e:
            self.node.get_logger().warning(f"Alpha shape failed: {e}")
            return 0.0, None

    # --------------------------------------------------------
    # MAIN ANALYSIS
    # --------------------------------------------------------
    def analyze(self, dataset_file=None): # NEW: Allow explicit dataset file passing
        try:
            self.node.get_logger().info("🚀 Starting Workspace Analysis...")
            
            # Load Data
            if dataset_file is None:
                dataset_file = self.load_latest_dataset()
                
            if dataset_file is None:
                return None

            df = pd.read_csv(dataset_file)

            if df.empty:
                self.node.get_logger().warning("⚠️ Dataset is empty. Aborting analysis.")
                return None

            if not {"x", "y", "z"}.issubset(df.columns):
                self.node.get_logger().error("❌ CSV missing required columns x,y,z. Aborting.")
                return None

            df = df.dropna(subset=["x", "y", "z"])
            points = df[["x", "y", "z"]].values
            
            # Safety Check: Needs at least 4 points for 3D Convex Hull
            if len(points) < 4:
                self.node.get_logger().error(f"❌ Not enough points for 3D convex hull (found {len(points)}, minimum 4 required).")
                return None

            self.node.get_logger().info(f"📊 Extracted {len(points)} valid 3D points. Calculating metrics...")

            metrics = {}

            # 1. Reach Radius Metrics
            self.node.get_logger().info("   ➜ Calculating Reach Radius Metrics...")
            r = np.linalg.norm(points, axis=1)
            metrics["max_reach_radius"] = float(np.max(r))
            metrics["min_reach_radius"] = float(np.min(r))

            # 2. Convex Hull & Volumes
            self.node.get_logger().info("   ➜ Computing Convex Hull & Reachable Surface Area...")
            hull = ConvexHull(points)
            metrics["workspace_volume"] = float(hull.volume)
            metrics["workspace_convex_hull_volume"] = float(hull.volume)
            metrics["reachable_surface_area"] = float(hull.area)

            # Concave hull(alpha shape)
            alpha_volume, alpha_mesh = self.compute_alpha_shape_volume(points, alpha=2.0)
            metrics["alpha_shape_volume"] = alpha_volume

            # 3. Workspace Bounding Box & Coverage
            self.node.get_logger().info("   ➜ Calculating Bounding Box Coverage...")
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            bbox_volume = np.prod(bbox_max - bbox_min)
            
            if bbox_volume > 0:
                metrics["workspace_coverage"] = float(hull.volume / bbox_volume)
            else:
                metrics["workspace_coverage"] = 0.0

            # 4. Reachability Density & Ratio
            self.node.get_logger().info("   ➜ Analyzing Voxel Density & Reachability Ratio...")
            voxel_size = 0.02
            voxel_index = np.floor(points / voxel_size).astype(int)
            unique_voxels = len(set(map(tuple, voxel_index)))
            metrics["reachability_density"] = unique_voxels / len(points)
            metrics["reachability_ratio"] = float(len(points) / max(unique_voxels, 1))

            # 5. Reachability Entropy
            self.node.get_logger().info("   ➜ Computing Reachability Entropy...")
            hist, _ = np.histogramdd(points, bins=20)
            prob = hist.flatten() / np.sum(hist)
            prob = prob[prob > 0]
            metrics["reachability_entropy"] = float(entropy(prob))

            # 6. Workspace Anisotropy
            self.node.get_logger().info("   ➜ Calculating Anisotropy via Covariance Eigenvalues...")
            cov = np.cov(points.T)
            eigvals = np.real(np.linalg.eigvals(cov))
            metrics["workspace_anisotropy"] = float(np.max(eigvals) / max(np.min(eigvals), 1e-9))

            # 7. Concave Hull (Alpha Shape Approximation)
            self.node.get_logger().info("   ➜ Computing Concave Hull Approximation (KDTree)...")
            tree = KDTree(points)
            neighbor_dist, _ = tree.query(points, k=5)
            alpha_approx_volume = np.mean(neighbor_dist)
            metrics["average_neighbor_distance"] = float(alpha_approx_volume)

            # 8. Dexterous & Orientation Workspace (Handling Quaternions)
            self.node.get_logger().info("   ➜ Checking for Dexterous/Orientation Workspace (Quaternions)...")
            if {"qx", "qy", "qz", "qw"}.issubset(df.columns):
                self.node.get_logger().info("      ↳ Quaternions found! Calculating Orientation Workspace...")
                df_quat = df.dropna(subset=["qx", "qy", "qz", "qw"])
                quats = df_quat[["qx", "qy", "qz", "qw"]].values
                
                # Convert Quaternions to Euler angles for variance calculation
                eulers = R.from_quat(quats).as_euler('xyz', degrees=True)
                orientation_variation = np.mean(np.std(eulers, axis=0))

                metrics["dexterous_workspace_volume"] = float(hull.volume * (orientation_variation / 180.0))
                metrics["orientation_workspace_volume"] = float(orientation_variation)

                # Generate Orientation Sphere Plot
                self.plot_orientation_sphere(quats)
            else:
                self.node.get_logger().info("      ↳ No orientation data found. Skipping dexterous metrics.")
                metrics["dexterous_workspace_volume"] = 0.0
                metrics["orientation_workspace_volume"] = 0.0

            # ------------------------------------------------
            # SAVE METRICS & DISTRIBUTIONS
            # ------------------------------------------------
            self.node.get_logger().info("💾 Saving overall metrics summary to CSV...")
            metrics_file = os.path.join(self.output_dir, "workspace_metrics_summary.csv")
            pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
            
            self.node.get_logger().info("💾 Saving statistical distributions for benchmark comparisons...")
            
            # 1. Radius Distribution
            pd.DataFrame({"reach_radius": r}).to_csv(os.path.join(self.distributions_dir, "radius_distribution.csv"), index=False)
            
            # 2. Voxel Distribution (Coordinates)
            pd.DataFrame(voxel_index, columns=["voxel_x", "voxel_y", "voxel_z"]).to_csv(os.path.join(self.distributions_dir, "voxel_distribution.csv"), index=False)
            
            # 3. Direction Distribution (Theta and Phi)
            theta = np.arctan2(points[:, 1], points[:, 0])
            pd.DataFrame({"theta_rad": theta}).to_csv(os.path.join(self.distributions_dir, "direction_distribution.csv"), index=False)
            
            self.node.get_logger().info(f"✅ Statistics & Distributions successfully saved!")

            # ------------------------------------------------
            # GENERATE PLOTS
            # ------------------------------------------------
            self.node.get_logger().info("📈 Generating and rendering all Workspace Plots. This may take a moment...")

            self.plot_3d_workspace(points)
            self.plot_convex_hull(points, hull)
            self.plot_alpha_shape(alpha_mesh)
            self.plot_density(df) 
            self.plot_radial_distribution(r)
            self.plot_polar_distribution(points)
            self.plot_voxel_occupancy_1d(points) 
            self.plot_voxel_occupancy_3d(points)  
            self.plot_workspace_slices(points)    
            self.plot_cross_section(points)       
            self.plot_workspace_projections(points)
            self.plot_directional_histogram(points)
            self.plot_surface_mesh(points, hull) 
            self.plot_metrics_bar_chart(metrics)

            self.node.get_logger().info("✅ Workspace Analysis and Plotting COMPLETED successfully.")
            return metrics

        except Exception as e:
            self.node.get_logger().error(f"❌ Workspace analysis failed: {str(e)}")
            return None

    # ------------------------------------------------
    # PLOTTING FUNCTIONS
    # ------------------------------------------------

    def plot_3d_workspace(self, points):
        self.node.get_logger().info("   ↳ Plotting 3D Workspace Scatter...")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:,0], points[:,1], points[:,2], s=1, alpha=0.5)
        ax.set_title("3D Workspace Scatter")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/workspace_3d_scatter.png")
        plt.close()

    def plot_convex_hull(self, points, hull):
        self.node.get_logger().info("   ↳ Plotting Convex Hull wireframe...")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for simplex in hull.simplices:
            ax.plot(points[simplex,0], points[simplex,1], points[simplex,2], "c-", alpha=0.5)
        ax.set_title("Workspace Convex Hull")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/workspace_convex_hull.png")
        plt.close()

    def plot_alpha_shape(self, mesh):
        if mesh is None:
            return

        self.node.get_logger().info("   ↳ Plotting Alpha Shape Surface...")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_trisurf(
            mesh.vertices[:,0],
            mesh.vertices[:,1],
            mesh.vertices[:,2],
            triangles=mesh.faces,
            cmap='viridis',
            alpha=0.8
        )

        ax.set_title("Concave Workspace (Alpha Shape)")

        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/alpha_shape_workspace.png")
        plt.close()

    def plot_surface_mesh(self, points, hull):
        self.node.get_logger().info("   ↳ Plotting Reachable Surface Mesh...")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices, cmap='viridis', alpha=0.7)
        ax.set_title("Reachable Workspace Surface Mesh")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/workspace_surface_mesh.png")
        plt.close()

    def plot_orientation_sphere(self, quats):
        self.node.get_logger().info("   ↳ Plotting Orientation Workspace Sphere...")
        vecs = R.from_quat(quats).apply([1, 0, 0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vecs[:,0], vecs[:,1], vecs[:,2], s=1, alpha=0.3, color='orange')
        ax.set_title("Orientation Workspace Sphere")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/orientation_sphere.png")
        plt.close()

    def plot_density(self, df):
        self.node.get_logger().info("   ↳ Plotting 2D Reachability Heatmaps (Density)...")
        pairs =[("x","y"),("x","z"),("y","z")]
        for a,b in pairs:
            plt.figure()
            plt.hist2d(df[a], df[b], bins=50, cmap='plasma')
            plt.colorbar(label='Point Count')
            plt.title(f"Workspace Density Heatmap {a.upper()}-{b.upper()}")
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/workspace_density_{a}{b}.png")
            plt.close()

    def plot_radial_distribution(self, r):
        self.node.get_logger().info("   ↳ Plotting Radial Distribution...")
        plt.figure()
        plt.hist(r, bins=50, color='skyblue', edgecolor='black')
        plt.title("Reachability Distribution (Max/Min Radius)")
        plt.xlabel("Radius (m)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/reachability_distribution.png")
        plt.close()

    def plot_polar_distribution(self, points):
        self.node.get_logger().info("   ↳ Plotting Polar Distribution (Anisotropy)...")
        r = np.linalg.norm(points[:, :2], axis=1)
        theta = np.arctan2(points[:,1], points[:,0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.scatter(theta, r, s=1, alpha=0.5)
        ax.set_title("Polar Reachability (Anisotropy)")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/polar_reachability.png")
        plt.close()

    def plot_voxel_occupancy_1d(self, points):
        self.node.get_logger().info("   ↳ Plotting 1D Voxel Occupancy (X-Axis)...")
        voxel = np.floor(points / 0.02)
        plt.figure()
        plt.hist(voxel[:,0], bins=50, color='green', alpha=0.7)
        plt.title("Voxel Occupancy (X-axis Projection)")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/voxel_occupancy_1d.png")
        plt.close()

    def plot_voxel_occupancy_3d(self, points):
        self.node.get_logger().info("   ↳ Plotting 3D Voxel Reachability Grid...")
        voxel_size = 0.02
        voxel_indices = np.floor(points / voxel_size).astype(int)
        unique_voxels = np.unique(voxel_indices, axis=0)
        
        voxel_centers = unique_voxels * voxel_size + (voxel_size / 2.0)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(voxel_centers[:,0], voxel_centers[:,1], voxel_centers[:,2], s=2, alpha=0.6, color='darkred')
        ax.set_title("3D Voxel Reachability Grid")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/voxel_occupancy_3d.png")
        plt.close()

    def plot_workspace_slices(self, points):
        self.node.get_logger().info("   ↳ Plotting Workspace Slices (Z=Constant)...")
        z_levels = np.linspace(points[:,2].min(), points[:,2].max(), 6)

        for z in z_levels:
            mask = np.abs(points[:,2] - z) < 0.02

            plt.figure()
            plt.scatter(points[mask,0], points[mask,1], s=2, color='teal')
            plt.title(f"Workspace Slice Z={z:.2f} m")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.grid(True, linestyle='--', alpha=0.6)
            
            safe_z = f"{z:.2f}".replace("-", "neg")
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/workspace_slice_{safe_z}.png")
            plt.close()

    def plot_cross_section(self, points):
        self.node.get_logger().info("   ↳ Plotting Cross Section (Radius vs Z)...")
        r_xy = np.linalg.norm(points[:, :2], axis=1)
        z = points[:, 2]

        plt.figure()
        plt.scatter(r_xy, z, s=1, alpha=0.3, color='magenta')
        plt.title("Workspace Cross Section (Radius vs Height Z)")
        plt.xlabel("Radius in XY Plane (m)")
        plt.ylabel("Height Z (m)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/workspace_cross_section.png")
        plt.close()

    def plot_workspace_projections(self, points):
        self.node.get_logger().info("   ↳ Plotting 2D Workspace Projections...")
        projections =[(0,1,"xy"),(0,2,"xz"),(1,2,"yz")]
        for a,b,name in projections:
            plt.figure()
            plt.scatter(points[:,a], points[:,b], s=1, alpha=0.3)
            plt.title(f"Reachable Area 2D Projection {name.upper()}")
            plt.xlabel(f"Axis {name[0].upper()}")
            plt.ylabel(f"Axis {name[1].upper()}")
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/workspace_projection_{name}.png")
            plt.close()

    def plot_directional_histogram(self, points):
        self.node.get_logger().info("   ↳ Plotting Directional Reachability Histogram...")
        theta = np.arctan2(points[:,1], points[:,0])
        plt.figure()
        plt.hist(theta, bins=50, color='purple', alpha=0.7)
        plt.title("Directional Reachability Histogram")
        plt.xlabel("Angle (Radians)")
        plt.ylabel("Point Count")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/directional_histogram.png")
        plt.close()

    def plot_metrics_bar_chart(self, metrics):
        self.node.get_logger().info("   ↳ Plotting Key Metrics Bar Chart...")
        keys_to_plot =["max_reach_radius", "workspace_volume", "reachability_ratio", 
                        "reachability_entropy", "workspace_coverage"]
        vals = [metrics[k] for k in keys_to_plot]
        
        plt.figure(figsize=(10, 6))
        plt.barh(keys_to_plot, vals, color='coral')
        plt.title("Key Workspace Metrics Summary")
        plt.xlabel("Value")
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/metrics_summary_barchart.png")
        plt.close()