import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import ConvexHull

class DexterityAnalyzer:

    def __init__(self, node, sampler, parser):
        self.node = node
        self.sampler = sampler     # Will use for Exact Jacobian / FK
        self.parser = parser

        self.joints = parser.joints
        self.dof = parser.dof

        self.output_dir = "dexterity_metrics"
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.dist_dir = os.path.join(self.output_dir, "distributions")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.dist_dir, exist_ok=True)

        self.node.get_logger().info("📂 DexterityAnalyzer initialized. Output directories verified.")

    # ------------------------------------------------
    # Load latest FK dataset
    # ------------------------------------------------
    def load_latest_dataset(self):
        self.node.get_logger().info("🔍 Searching for latest FK dataset...")
        csv_files = glob.glob("workspace_fk_dataset_*.csv")

        if not csv_files:
            self.node.get_logger().error("❌ No FK dataset found. Run Module 2 first.")
            return None

        latest = max(csv_files, key=os.path.getctime)
        self.node.get_logger().info(f"📁 Using dataset for Dexterity Analysis: {latest}")
        return latest

    # ------------------------------------------------
    # Jacobian Computation (Analytic + Central Diff Fallback)
    # ------------------------------------------------
    def compute_jacobian(self, q, delta=1e-3):
        """
        Attempts to use Exact Analytic Jacobian from MoveIt.
        Falls back to highly accurate Central Difference Numerical Jacobian.
        """
        # 1. Try Exact Analytic Jacobian (10x faster, perfectly accurate)
        if hasattr(self.sampler, "get_jacobian"):
            try:
                J_exact = self.sampler.get_jacobian(q)
                if J_exact is not None:
                    # MoveIt returns 6xN (Linear + Angular). We extract 3xN Linear for positional dexterity.
                    return J_exact[:3, :]
            except Exception as e:
                self.node.get_logger().debug(f"Analytic Jacobian failed, using fallback: {e}")

        # 2. Fallback: Central Difference Numerical Jacobian (High Accuracy)
        n = len(q)
        J = np.zeros((3, n))
        
        for i in range(n):
            q_plus = q.copy()
            q_minus = q.copy()
            
            q_plus[i] += delta
            q_minus[i] -= delta
            
            pose_plus = self.sampler.compute_fk(q_plus)
            pose_minus = self.sampler.compute_fk(q_minus)
            
            if pose_plus is None or pose_minus is None:
                return None
                
            p_plus = np.array([pose_plus.position.x, pose_plus.position.y, pose_plus.position.z])
            p_minus = np.array([pose_minus.position.x, pose_minus.position.y, pose_minus.position.z])
            
            # Central difference formula
            J[:, i] = (p_plus - p_minus) / (2 * delta)

        return J

    # ------------------------------------------------
    # Manipulability
    # ------------------------------------------------
    def manipulability(self, J):
        JJ = J @ J.T
        det = np.linalg.det(JJ)
        if det < 0:
            return 0
        return np.sqrt(det)

    # ------------------------------------------------
    # Condition Number & SVD components
    # ------------------------------------------------
    def condition_number(self, J):
        U, S, V = np.linalg.svd(J)
        smax = np.max(S)
        smin = np.min(S)

        if smin < 1e-9:
            return np.inf, smin, smax, U, S

        return smax / smin, smin, smax, U, S

    # ------------------------------------------------
    # MAIN ANALYSIS
    # ------------------------------------------------
    def analyze(self):
        try:
            dataset = self.load_latest_dataset()
            if dataset is None:
                return None

            df = pd.read_csv(dataset)
            if df.empty:
                self.node.get_logger().error("❌ Dataset empty.")
                return None

            df = df.dropna(subset=["x", "y", "z"])
            self.node.get_logger().info(f"📊 Dataset size: {len(df)} valid samples ready for analysis.")

            manipulability_vals =[]
            cond_vals = []
            sigma_min_vals = []
            sigma_max_vals =[]
            valid_points = []
            
            U_matrices = []
            S_matrices =[]

            self.node.get_logger().info("🚀 Computing dexterity metrics...")

            # PERFORMANCE FIX: Direct NumPy array iteration (100x faster than iterrows/iloc)
            q_matrix = df[self.joints].values
            points_matrix = df[["x", "y", "z"]].values

            for idx in range(len(q_matrix)):
                q = q_matrix[idx]
                J = self.compute_jacobian(q)

                if J is None:
                    continue

                w = self.manipulability(J)
                cond, smin, smax, U, S = self.condition_number(J)

                manipulability_vals.append(w)
                cond_vals.append(cond)
                sigma_min_vals.append(smin)
                sigma_max_vals.append(smax)
                valid_points.append(points_matrix[idx])
                
                U_matrices.append(U)
                S_matrices.append(S)

                if idx > 0 and idx % 1000 == 0:
                    self.node.get_logger().info(f"   ➜ Processed {idx} configurations...")

            if len(valid_points) == 0:
                self.node.get_logger().error("❌ No valid points left for dexterity analysis.")
                return None

            points = np.array(valid_points)
            manipulability_vals = np.array(manipulability_vals)
            cond_vals = np.array(cond_vals)
            sigma_min_vals = np.array(sigma_min_vals)
            sigma_max_vals = np.array(sigma_max_vals)

            # ------------------------------------------------
            # Advanced Index Calculations
            # ------------------------------------------------
            self.node.get_logger().info("🧮 Calculating Advanced Indices (Isotropy, KCI, LDI, GDI, WCI)...")
            
            isotropy_vals = np.where(cond_vals == np.inf, 0, 1.0 / cond_vals)
            kci_vals = isotropy_vals.copy() 
            
            max_w = np.max(manipulability_vals) if np.max(manipulability_vals) > 0 else 1
            ldi_vals = manipulability_vals / max_w
            gdi_value = np.mean(ldi_vals)

            # SCIENTIFIC ADDITION: Workspace Coverage Index (WCI)
            try:
                hull = ConvexHull(points)
                reachable_volume = hull.volume
                bbox_min = points.min(axis=0)
                bbox_max = points.max(axis=0)
                bounding_volume = np.prod(bbox_max - bbox_min)
                wci = reachable_volume / bounding_volume if bounding_volume > 0 else 0
            except Exception as e:
                self.node.get_logger().warning(f"Could not compute WCI: {e}")
                wci = 0.0

            # ------------------------------------------------
            # Metrics Mapping
            # ------------------------------------------------
            metrics = {}
            metrics["manipulability_min"] = float(np.min(manipulability_vals))
            metrics["manipulability_max"] = float(np.max(manipulability_vals))
            metrics["manipulability_mean"] = float(np.mean(manipulability_vals))
            metrics["manipulability_variance"] = float(np.var(manipulability_vals))
            
            # SAFETY FIX: Handle np.inf gracefully
            finite_cond = cond_vals[np.isfinite(cond_vals)]
            metrics["condition_number_mean"] = float(np.mean(finite_cond)) if len(finite_cond) > 0 else float("inf")
            
            metrics["isotropy_mean"] = float(np.mean(isotropy_vals))
            metrics["kci_mean"] = float(np.mean(kci_vals))
            metrics["ldi_mean"] = float(np.mean(ldi_vals))
            metrics["gdi"] = float(gdi_value)
            metrics["workspace_coverage_index"] = float(wci)

            singular = sigma_min_vals < 0.01
            metrics["singularity_ratio"] = float(np.sum(singular) / len(sigma_min_vals))

            # ------------------------------------------------
            # Save metrics & Distributions
            # ------------------------------------------------
            self.node.get_logger().info("💾 Saving Dexterity Metrics & Distributions...")
            pd.DataFrame([metrics]).to_csv(os.path.join(self.output_dir, "dexterity_metrics_summary.csv"), index=False)
            pd.DataFrame({"manipulability": manipulability_vals}).to_csv(os.path.join(self.dist_dir, "manipulability_distribution.csv"), index=False)
            pd.DataFrame({"condition_number": cond_vals}).to_csv(os.path.join(self.dist_dir, "condition_number_distribution.csv"), index=False)
            pd.DataFrame({"sigma_min": sigma_min_vals}).to_csv(os.path.join(self.dist_dir, "singularity_distance_distribution.csv"), index=False)
            pd.DataFrame({"isotropy": isotropy_vals, "ldi": ldi_vals}).to_csv(os.path.join(self.dist_dir, "advanced_indices.csv"), index=False)

            # ------------------------------------------------
            # PLOTS RENDERING
            # ------------------------------------------------
            self.node.get_logger().info("📈 Generating all requested plots...")

            self.plot_manipulability_hist(manipulability_vals)
            self.plot_manipulability_heatmap_2d(points, manipulability_vals)
            self.plot_manipulability_ellipsoids(points, U_matrices, S_matrices)
            self.plot_manipulability_stats_bar(metrics)
            self.plot_boxplot(manipulability_vals)
            self.plot_condition_heatmap(points, cond_vals)
            self.plot_isotropy_contour(points, isotropy_vals)
            self.plot_kci_heatmap(points, kci_vals)
            self.plot_ldi_spatial_heatmap(points, ldi_vals)
            self.plot_gdi_radar(metrics)
            self.plot_singularity_map(points, sigma_min_vals)
            self.plot_singularity_hist(sigma_min_vals)
            self.plot_singularity_ratio_bar(metrics)

            self.node.get_logger().info("✅ All Dexterity Analysis & Plotting Successfully Completed.")
            return metrics

        except Exception as e:
            self.node.get_logger().error(f"❌ Dexterity analysis failed: {str(e)}")
            return None

    # ================================================
    # PLOTTING FUNCTIONS
    # ================================================

    def plot_manipulability_hist(self, vals):
        plt.figure()
        plt.hist(vals, bins=50, color='teal', edgecolor='black')
        plt.title("Manipulability Index Distribution")
        plt.xlabel("Manipulability (w)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/manipulability_histogram.png")
        plt.close()

    def plot_manipulability_heatmap_2d(self, points, vals):
        plt.figure()
        plt.scatter(points[:,0], points[:,1], c=vals, cmap='viridis', s=3)
        plt.colorbar(label="Manipulability Index (w)")
        plt.title("Manipulability Heatmap (XY Projection)")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/manipulability_heatmap.png")
        plt.close()

    def plot_manipulability_ellipsoids(self, points, U_matrices, S_matrices):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        
        step = max(1, len(points) // 25)
        
        # VISUALIZATION FIX: Compute base sphere ONCE outside the loop
        u = np.linspace(0, 2 * np.pi, 12)
        v = np.linspace(0, np.pi, 12)
        x_base = np.outer(np.cos(u), np.sin(v))
        y_base = np.outer(np.sin(u), np.sin(v))
        z_base = np.outer(np.ones_like(u), np.cos(v))
        sphere_coords = np.vstack((x_base.flatten(), y_base.flatten(), z_base.flatten()))

        for i in range(0, len(points), step):
            p = points[i]
            U = U_matrices[i]
            S = S_matrices[i]
            
            transform = U @ np.diag(S * 0.05)
            ellipsoid = transform @ sphere_coords
            
            x_el = ellipsoid[0, :].reshape(x_base.shape) + p[0]
            y_el = ellipsoid[1, :].reshape(y_base.shape) + p[1]
            z_el = ellipsoid[2, :].reshape(z_base.shape) + p[2]
            
            ax.plot_wireframe(x_el, y_el, z_el, color='blue', alpha=0.3)
            
        ax.set_title("Manipulability Ellipsoid Visualization")
        plt.savefig(f"{self.plots_dir}/manipulability_ellipsoids.png")
        plt.close()

    def plot_manipulability_stats_bar(self, metrics):
        plt.figure()
        labels =['Minimum', 'Average', 'Maximum']
        vals = [metrics["manipulability_min"], metrics["manipulability_mean"], metrics["manipulability_max"]]
        
        plt.bar(labels, vals, color=['#ff9999','#66b3ff','#99ff99'])
        plt.title("Manipulability Index Statistics")
        plt.ylabel("Manipulability Value")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/manipulability_stats_barchart.png")
        plt.close()

    def plot_boxplot(self, vals):
        plt.figure()
        plt.boxplot(vals, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.title("Manipulability Variance (Box Plot)")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/manipulability_boxplot.png")
        plt.close()

    def plot_condition_heatmap(self, points, cond):
        plt.figure()
        
        # SAFETY FIX: Prevent crash when calculating percentile of array with all infs
        finite_cond = cond[np.isfinite(cond)]
        cap = np.percentile(finite_cond, 95) if len(finite_cond) > 0 else 1
        c_capped = np.clip(cond, 0, cap)

        plt.scatter(points[:,0], points[:,1], c=c_capped, cmap='plasma', s=3)
        plt.colorbar(label="Condition Number (Capped at 95th Percentile)")
        plt.title("Jacobian Conditioning Heatmap")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/condition_number_heatmap.png")
        plt.close()

    def plot_isotropy_contour(self, points, isotropy):
        plt.figure()
        try:
            plt.tricontourf(points[:,0], points[:,1], isotropy, levels=20, cmap='YlGnBu')
            plt.colorbar(label="Isotropy Index (0 to 1)")
            plt.title("Isotropy Index Contour Plot")
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/isotropy_contour.png")
        except Exception as e:
            self.node.get_logger().warning(f"Could not generate Isotropy Contour: {e}")
        plt.close()

    def plot_kci_heatmap(self, points, kci):
        plt.figure()
        plt.scatter(points[:,0], points[:,2], c=kci, cmap='magma', s=3)
        plt.colorbar(label="KCI")
        plt.title("Kinematic Conditioning Index (XZ Projection)")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/kci_heatmap.png")
        plt.close()

    def plot_ldi_spatial_heatmap(self, points, ldi):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(points[:,0], points[:,1], points[:,2], c=ldi, cmap='coolwarm', s=3)
        plt.colorbar(sc, label="Local Dexterity Index (LDI)")
        plt.title("Local Dexterity Index Spatial Heatmap")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/ldi_spatial_heatmap.png")
        plt.close()

    def plot_gdi_radar(self, metrics):
        labels = np.array(['GDI', 'Avg Isotropy', 'Avg KCI', 'Safe Workspace Ratio'])
        safe_ratio = 1.0 - metrics["singularity_ratio"]
        stats = np.array([metrics["gdi"], metrics["isotropy_mean"], metrics["kci_mean"], safe_ratio])
        
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        stats = np.concatenate((stats, [stats[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.fill(angles, stats, color='green', alpha=0.25)
        ax.plot(angles, stats, color='green', linewidth=2)
        ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
        
        plt.title("Global Dexterity Overview (Radar Chart)", y=1.08)
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/global_dexterity_radar.png")
        plt.close()

    def plot_singularity_map(self, points, smin):
        mask = smin < 0.01
        plt.figure()
        plt.scatter(points[:,0], points[:,1], c='lightgrey', s=2, alpha=0.5, label='Safe')
        plt.scatter(points[mask,0], points[mask,1], color="red", s=8, label='Singular (σ_min < 0.01)')
        plt.title("Singularity Proximity Map")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/singularity_map.png")
        plt.close()

    def plot_singularity_hist(self, smin):
        plt.figure()
        plt.hist(smin, bins=50, color='orange', edgecolor='black')
        plt.title("Distance to Singularity (Minimum Singular Value)")
        plt.xlabel("σ_min")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/singularity_distance_histogram.png")
        plt.close()

    def plot_singularity_ratio_bar(self, metrics):
        plt.figure()
        ratios = [metrics["singularity_ratio"], 1.0 - metrics["singularity_ratio"]]
        labels = ["Singular States", "Safe States"]
        colors = ["red", "green"]
        
        plt.bar(labels, ratios, color=colors)
        plt.title("Workspace Singularity Ratio")
        plt.ylabel("Ratio (Percentage)")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/singularity_ratio_barchart.png")
        plt.close()