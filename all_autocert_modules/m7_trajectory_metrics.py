import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Ignore division by zero warnings for complex path calculations
warnings.filterwarnings("ignore", category=RuntimeWarning)

class TrajectoryPerformanceAnalyzer:

    def __init__(self, node, planner_interface):
        self.node = node
        self.planner = planner_interface # MUST have plan_to_pose(pose) AND compute_fk(joint_config)

        # ------------------------------------------------
        # DIRECTORY SETUP
        # ------------------------------------------------
        self.output_dir = "trajectory_metrics"
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.dist_dir = os.path.join(self.output_dir, "distributions")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.dist_dir, exist_ok=True)

        self.node.get_logger().info("? TrajectoryPerformanceAnalyzer (Module 6) initialized.")
        self.node.get_logger().info("? Ready to analyze kinematic and dynamic path quality.")

    # ------------------------------------------------
    # Load dataset
    # ------------------------------------------------
    def load_latest_dataset(self):
        self.node.get_logger().info("? Searching for the latest FK dataset...")
        csv_files = glob.glob("workspace_fk_dataset_*.csv")

        if not csv_files:
            self.node.get_logger().error("? No FK dataset found! Run Module 2 first.")
            return None

        latest = max(csv_files, key=os.path.getctime)
        self.node.get_logger().info(f"? Using dataset for trajectory evaluation: {latest}")
        return latest

    # ------------------------------------------------
    # MAIN ANALYSIS PIPELINE
    # ------------------------------------------------
    def analyze(self, max_samples=1000):
        try:
            dataset = self.load_latest_dataset()
            if dataset is None: return None

            self.node.get_logger().info("⏳ Loading dataset into memory...")
            df = pd.read_csv(dataset)
            
            poses = df[["x", "y", "z"]].values
            joint_targets = df[self.planner.joints].values
            total_poses = len(poses)

            if total_poses > max_samples:
                self.node.get_logger().info(f"📉 Downsampling to {max_samples} poses...")
                indices = np.random.choice(total_poses, max_samples, replace=False)
                poses = poses[indices]
                joint_targets = joint_targets[indices]
                total_poses = max_samples

            success, joint_path_lengths, cart_path_lengths, path_efficiencies = [], [], [],[]
            curvatures, smoothness_vals, max_jerks, max_accels, max_vels = [], [], [], [],[]
            motion_energies, motion_costs, durations, time_optimalities, path_deviations = [], [], [], [],[]

            rep_traj = None 

            self.node.get_logger().info("🚀 ========================================================")
            self.node.get_logger().info("🚀 STARTING TRAJECTORY & PATH QUALITY BENCHMARKING")
            self.node.get_logger().info("🚀 ========================================================")

            for i, (pose, joint_target) in enumerate(zip(poses, joint_targets)):
                
                result = self.planner.plan_to_joint_target(joint_target)

                if result is None or "trajectory" not in result or len(result["trajectory"]) < 2:
                    success.append(0)
                    continue

                success.append(1)
                traj = result["trajectory"]
                q = np.array([pt.get("positions", [0]*6) for pt in traj]) 
                t = np.array([pt.get("time_from_start", idx*0.1) for idx, pt in enumerate(traj)])
                
                dt = np.diff(t)
                dt[dt == 0] = 1e-6 

                duration = t[-1] - t[0]
                durations.append(duration)

                dq = np.diff(q, axis=0)
                joint_length = np.sum(np.linalg.norm(dq, axis=1))
                joint_path_lengths.append(joint_length)

                v = dq / dt[:, np.newaxis]
                a = np.diff(v, axis=0) / dt[1:, np.newaxis]
                jerk = np.diff(a, axis=0) / dt[2:, np.newaxis]

                max_v = np.max(np.linalg.norm(v, axis=1)) if len(v) > 0 else 0
                max_a = np.max(np.linalg.norm(a, axis=1)) if len(a) > 0 else 0
                max_j = np.max(np.linalg.norm(jerk, axis=1)) if len(jerk) > 0 else 0
                
                max_vels.append(max_v)
                max_accels.append(max_a)
                max_jerks.append(max_j)

                smoothness = np.sum(np.linalg.norm(jerk, axis=1)**2 * dt[2:]) if len(jerk) > 0 else 0
                smoothness_vals.append(smoothness)

                energy = np.sum(np.linalg.norm(v, axis=1)**2 * dt) if len(v) > 0 else 0
                motion_energies.append(energy)

                cart_positions =[]
                for joint_config in q:
                    pose_fk = self.planner.compute_fk(joint_config)
                    if pose_fk:
                        cart_positions.append([pose_fk.position.x, pose_fk.position.y, pose_fk.position.z])
                    else:
                        cart_positions.append(cart_positions[-1] if len(cart_positions) > 0 else [0, 0, 0])
                
                cart_positions = np.array(cart_positions)

                if len(cart_positions) > 1:
                    dp = np.diff(cart_positions, axis=0)
                    actual_cart_dist = np.sum(np.linalg.norm(dp, axis=1))
                    ideal_cart_dist = np.linalg.norm(pose) # Pose from CSV
                else:
                    actual_cart_dist, ideal_cart_dist = 0, 0

                cart_path_lengths.append(actual_cart_dist)
                path_efficiencies.append(ideal_cart_dist / max(actual_cart_dist, 1e-6))
                path_deviations.append(abs(actual_cart_dist - ideal_cart_dist))

                if len(a) > 0 and len(v) > 1:
                    v_align = v[1:] 
                    v_norm_sq = np.linalg.norm(v_align, axis=1)**2 + 1e-6
                    cross_term = np.linalg.norm(np.cross(v_align[:, :3], a[:, :3]), axis=1) if q.shape[1] >= 3 else 0
                    kappa = np.mean(cross_term / (v_norm_sq**1.5))
                    curvatures.append(kappa)
                else:
                    curvatures.append(0)

                ideal_time = joint_length / max(max_v, 0.1)
                time_optimalities.append(ideal_time / max(duration, 1e-6))

                cost = (0.4 * duration) + (0.4 * joint_length) + (0.2 * energy)
                motion_costs.append(cost)

                if rep_traj is None and len(traj) > 10:
                    rep_traj = {'t': t, 'v': v, 'a': a, 'j': jerk}

                if (i + 1) % max(1, total_poses // 20) == 0:
                    self.node.get_logger().info(f"⏳ Processed {i+1}/{total_poses} paths ({(i+1)/total_poses*100:.1f}%)")

            self.node.get_logger().info("🧮 Compiling and saving trajectory distributions...")

            def safe_mean(lst): return float(np.mean(lst)) if len(lst) > 0 else 0.0

            metrics = {
                "success_rate": safe_mean(success),
                "avg_joint_path_length": safe_mean(joint_path_lengths),
                "avg_cartesian_path_length": safe_mean(cart_path_lengths),
                "avg_path_efficiency": safe_mean(path_efficiencies),
                "avg_curvature": safe_mean(curvatures),
                "avg_smoothness": safe_mean(smoothness_vals),
                "avg_max_jerk": safe_mean(max_jerks),
                "avg_max_acceleration": safe_mean(max_accels),
                "avg_motion_energy": safe_mean(motion_energies),
                "avg_motion_cost": safe_mean(motion_costs),
                "avg_trajectory_duration": safe_mean(durations),
                "avg_time_optimality": safe_mean(time_optimalities)
            }

            pd.DataFrame([metrics]).to_csv(os.path.join(self.output_dir, "trajectory_metrics_summary.csv"), index=False)

            pd.DataFrame({
                "duration": durations,
                "joint_length": joint_path_lengths,
                "cart_length": cart_path_lengths,
                "efficiency": path_efficiencies,
                "curvature": curvatures,
                "smoothness": smoothness_vals,
                "jerk": max_jerks,
                "energy": motion_energies,
                "cost": motion_costs,
                "deviation": path_deviations
            }).to_csv(os.path.join(self.dist_dir, "trajectory_distributions.csv"), index=False)

            self.node.get_logger().info("📈 Generating Trajectory & Path Quality Plots...")
            
            self.plot_path_lengths(metrics)
            if len(path_efficiencies) > 0: self.plot_path_efficiency(path_efficiencies)
            if len(curvatures) > 0: self.plot_curvature_hist(curvatures)
            if len(smoothness_vals) > 0: self.plot_smoothness_curve(smoothness_vals)
            self.plot_energy_and_cost(metrics)
            if len(durations) > 0: self.plot_duration_hist(durations)
            if len(time_optimalities) > 0: self.plot_time_optimality(time_optimalities)
            if len(path_deviations) > 0: self.plot_path_deviation(path_deviations)

            if rep_traj is not None:
                self.plot_velocity_profile(rep_traj['t'], rep_traj['v'])
                self.plot_acceleration_profile(rep_traj['t'], rep_traj['a'])
                self.plot_jerk_profile(rep_traj['t'], rep_traj['j'])

            self.node.get_logger().info("✅ Module 7 Completed Successfully!")
            return metrics

        except Exception as e:
            self.node.get_logger().error(f"❌ Trajectory analysis failed: {e}")
            return None
    # ------------------------------------------------
    # PLOT GENERATION FUNCTIONS
    # ------------------------------------------------

    def plot_path_lengths(self, metrics):
        plt.figure(figsize=(6, 5))
        vals = [metrics["avg_joint_path_length"], metrics["avg_cartesian_path_length"]]
        plt.bar(["Joint Space (rad)", "Cartesian Space (m)"], vals, color=['#3498db', '#2ecc71'])
        plt.title("Average Path Lengths")
        plt.ylabel("Distance")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{self.plots_dir}/path_lengths_bar.png")
        plt.close()

    def plot_path_efficiency(self, efficiencies):
        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(efficiencies)), efficiencies, s=3, color='#9b59b6', alpha=0.5)
        plt.title("Path Efficiency (Ideal Length / Actual Length)")
        plt.xlabel("Trajectory ID")
        plt.ylabel("Efficiency Ratio (1.0 = Perfect Straight Line)")
        plt.grid(linestyle='--', alpha=0.5)
        plt.savefig(f"{self.plots_dir}/path_efficiency_scatter.png")
        plt.close()

    def plot_curvature_hist(self, curvatures):
        plt.figure(figsize=(8, 5))
        plt.hist(curvatures, bins=40, color='#e67e22', edgecolor='black')
        plt.title("Path Curvature Distribution")
        plt.xlabel("Average Curvature (k)")
        plt.ylabel("Frequency")
        plt.savefig(f"{self.plots_dir}/path_curvature_plot.png")
        plt.close()

    def plot_smoothness_curve(self, smoothness):
        sorted_vals = np.sort(smoothness)
        plt.figure(figsize=(8, 5))
        plt.plot(np.linspace(0, 100, len(sorted_vals)), sorted_vals, color='#1abc9c', linewidth=2)
        plt.title("Path Smoothness Curve (Cumulative Profile)")
        plt.xlabel("Percentile (%)")
        plt.ylabel("Smoothness Cost (Integral of Squared Jerk)")
        plt.grid(True)
        plt.savefig(f"{self.plots_dir}/path_smoothness_curve.png")
        plt.close()

    def plot_energy_and_cost(self, metrics):
        plt.figure(figsize=(7, 5))
        vals = [metrics["avg_motion_energy"], metrics["avg_motion_cost"]]
        plt.bar(["Motion Energy", "Motion Cost"], vals, color=['#f1c40f', '#e74c3c'])
        plt.title("Motion Energy & Cost Breakdown")
        plt.ylabel("Magnitude")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{self.plots_dir}/motion_energy_cost_bar.png")
        plt.close()

    def plot_duration_hist(self, durations):
        plt.figure(figsize=(8, 5))
        plt.hist(durations, bins=40, color='#34495e', edgecolor='white')
        plt.title("Trajectory Duration Histogram")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")
        plt.savefig(f"{self.plots_dir}/trajectory_duration_hist.png")
        plt.close()

    def plot_time_optimality(self, optimalities):
        plt.figure(figsize=(8, 5))
        plt.plot(optimalities, '.', color='#e84393', alpha=0.6, markersize=4)
        plt.title("Time Optimality Efficiency Plot")
        plt.xlabel("Trajectory ID")
        plt.ylabel("Optimality Ratio (Ideal Time / Actual Time)")
        plt.axhline(y=1.0, color='r', linestyle='--')
        plt.grid(True)
        plt.savefig(f"{self.plots_dir}/time_optimality_plot.png")
        plt.close()

    def plot_path_deviation(self, deviations):
        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(deviations)), deviations, color='#d35400', s=3, alpha=0.6)
        plt.title("Path Deviation Scatter Plot")
        plt.xlabel("Trajectory ID")
        plt.ylabel("Deviation from straight line (m)")
        plt.grid(True)
        plt.savefig(f"{self.plots_dir}/path_deviation_scatter.png")
        plt.close()

    def plot_velocity_profile(self, t, v):
        plt.figure(figsize=(8, 5))
        v_mag = np.linalg.norm(v, axis=1)
        t_plot = t[1:] 
        plt.plot(t_plot, v_mag, color='blue', linewidth=2)
        plt.title("Velocity Profile vs Time (Representative Trajectory)")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity Magnitude (rad/s)")
        plt.grid(True)
        plt.fill_between(t_plot, v_mag, color='blue', alpha=0.1)
        plt.savefig(f"{self.plots_dir}/velocity_vs_time_plot.png")
        plt.close()

    def plot_acceleration_profile(self, t, a):
        plt.figure(figsize=(8, 5))
        a_mag = np.linalg.norm(a, axis=1)
        t_plot = t[2:] 
        plt.plot(t_plot, a_mag, color='red', linewidth=2)
        plt.title("Acceleration Magnitude vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (rad/s²)")
        plt.grid(True)
        plt.savefig(f"{self.plots_dir}/acceleration_plot.png")
        plt.close()

    def plot_jerk_profile(self, t, j):
        plt.figure(figsize=(8, 5))
        j_mag = np.linalg.norm(j, axis=1)
        t_plot = t[3:] 
        plt.plot(t_plot, j_mag, color='purple', linewidth=2)
        plt.title("Jerk Magnitude vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Jerk (rad/s³)")
        plt.grid(True)
        plt.savefig(f"{self.plots_dir}/jerk_vs_time_plot.png")
        plt.close()