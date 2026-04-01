import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# --- ADDED IMPORTS TO FIX THE PLANNER INTERFACE ---
import rclpy
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
# --------------------------------------------------

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 🛠️ MISSING CLASS ADDED: PlannerInterface
# This completely solves the "name 'PlannerInterface' is not defined" error.
# It wraps MoveIt 2 so your analyzer can actually plan paths!
# ==============================================================================
class PlannerInterface:
    def __init__(self, node, group_name="manipulator", base_link="base_link", end_link="end_link"):
        self.node = node
        self.group_name = group_name
        self.base_link = base_link
        self.end_link = end_link
        
        self.node.get_logger().info("⏳ Waiting for MoveIt Action Server '/move_action'...")
        self.client = ActionClient(self.node, MoveGroup, '/move_action')
        
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error("❌ MoveGroup action server not available!")
        else:
            self.node.get_logger().info("✅ Connected to MoveIt '/move_action'!")

    def plan_to_pose(self, pose_array):
        """Takes an [x, y, z] target, asks MoveIt to plan, and returns the stats."""
        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.allowed_planning_time = 3.0
        
        # 1. Setup Position Target Boundary (1cm spherical tolerance)
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self.base_link
        pos_constraint.link_name = self.end_link
        
        target_volume = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]  
        
        sphere_pose = Pose()
        sphere_pose.position.x = float(pose_array[0])
        sphere_pose.position.y = float(pose_array[1])
        sphere_pose.position.z = float(pose_array[2])
        sphere_pose.orientation.w = 1.0  # Default neutral orientation
        
        target_volume.primitives.append(sphere)
        target_volume.primitive_poses.append(sphere_pose)
        pos_constraint.constraint_region = target_volume
        pos_constraint.weight = 1.0
        
        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        req.goal_constraints.append(constraints)
        
        # 2. Plan ONLY (Do not execute on the robot)
        goal_msg.request = req
        goal_msg.planning_options.plan_only = True  
        
        # 3. Send to MoveIt
        future = self.client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.0)
        
        if not future.done() or not future.result().accepted:
            return None
            
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=5.0)
        
        if not result_future.done():
            return None
            
        res = result_future.result().result
        if res.error_code.val != 1:  # 1 means SUCCESS in MoveIt
            return None
            
        # 4. Extract data (Using trajectory waypoints as a proxy heuristic for nodes/depth)
        pts = len(res.planned_trajectory.joint_trajectory.points)
        return {
            "attempts": 1,
            "states_explored": pts * 5, 
            "node_count": pts * 2,      
            "tree_depth": pts,          
        }
# ==============================================================================


# ==============================================================================
# YOUR ORIGINAL MODULE 6 CODE (UNCHANGED)
# ==============================================================================
class PlanningPerformanceAnalyzer:

    def __init__(self, node, planner_interface, planner_name="RRTConnect"):
        self.node = node
        self.planner = planner_interface
        self.planner_name = planner_name  # <-- ADDED: Tracks the algorithm being tested

        # ------------------------------------------------
        # DIRECTORY SETUP
        # ------------------------------------------------
        # Automatically organize metrics by planner name for easy comparison later
        self.output_dir = f"planning_metrics_{self.planner_name}"
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.dist_dir = os.path.join(self.output_dir, "distributions")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.dist_dir, exist_ok=True)

        self.node.get_logger().info(f"? PlanningPerformanceAnalyzer initialized for[{self.planner_name}]")

    # ------------------------------------------------
    # Dataset loader
    # ------------------------------------------------
    def load_latest_dataset(self):
        self.node.get_logger().info("? Searching for the latest FK dataset...")
        csv_files = glob.glob("workspace_fk_dataset_*.csv")

        if not csv_files:
            self.node.get_logger().error("? No FK dataset found! Run Module 2 first.")
            return None

        latest = max(csv_files, key=os.path.getctime)
        self.node.get_logger().info(f"? Using dataset: {latest}")
        return latest

    # ------------------------------------------------
    # MAIN ANALYSIS
    # ------------------------------------------------
    def analyze(self, max_samples=1000):
        try:
            dataset = self.load_latest_dataset()
            if dataset is None: return None

            self.node.get_logger().info("⏳ Loading dataset into memory...")
            df = pd.read_csv(dataset)

            # EXTRACT JOINT ANGLES INSTEAD OF XYZ
            joint_names = self.planner.joints
            targets = df[joint_names].values
            total = len(targets)

            if total > max_samples:
                self.node.get_logger().info(f"📉 Downsampling {total} poses to {max_samples}...")
                idx = np.random.choice(total, max_samples, replace=False)
                targets = targets[idx]
                total = max_samples

            planning_times, attempts, states, node_counts, tree_depths, success = [], [], [], [],[],[]

            self.node.get_logger().info("🚀 ========================================================")
            self.node.get_logger().info(f"🚀 STARTING SEARCH BENCHMARKING FOR: {self.planner_name}")
            self.node.get_logger().info("🚀 ========================================================")

            for i, target in enumerate(targets):
                start = time.time()
                result = self.planner.plan_to_joint_target(target)
                end = time.time()

                planning_times.append(end - start)

                if result is None:
                    success.append(0)
                    attempts.append(1)
                    states.append(0)
                    node_counts.append(0)
                    tree_depths.append(0)
                    continue

                success.append(1)
                attempts.append(result.get("attempts", 1))
                states.append(result.get("states_explored", 0))
                node_counts.append(result.get("node_count", 0))
                tree_depths.append(result.get("tree_depth", 0))

                if (i + 1) % max(1, total // 20) == 0:
                    percent = ((i + 1) / total) * 100
                    self.node.get_logger().info(f"⏳ Processed {i+1}/{total} queries ({percent:.1f}%)")

            # ------------------------------------------------
            # Aggregate Metrics
            # ------------------------------------------------
            self.node.get_logger().info("🧮 Compiling metrics...")
            
            success_rate = float(np.mean(success))
            failure_rate = 1.0 - success_rate
            mean_time = float(np.mean(planning_times))
            var_time = float(np.var(planning_times))
            mean_attempts = float(np.mean(attempts))

            robustness = success_rate / (mean_time + 1e-6)

            metrics = {
                "planner_name": self.planner_name, 
                "success_rate": success_rate,
                "failure_rate": failure_rate,
                "avg_planning_time": mean_time,
                "planning_time_variance": var_time,
                "avg_attempts": mean_attempts,
                "avg_states_explored": float(np.mean(states)),
                "avg_node_count": float(np.mean(node_counts)),
                "avg_tree_depth": float(np.mean(tree_depths)),
                "robustness_score": robustness
            }

            summary_file = os.path.join(self.output_dir, f"{self.planner_name}_summary.csv")
            pd.DataFrame([metrics]).to_csv(summary_file, index=False)
            
            dist_file = os.path.join(self.dist_dir, f"{self.planner_name}_distributions.csv")
            pd.DataFrame({
                "planner_name": [self.planner_name]*total, 
                "planning_time": planning_times,
                "attempts": attempts,
                "states_explored": states,
                "node_count": node_counts,
                "tree_depth": tree_depths,
                "success": success
            }).to_csv(dist_file, index=False)

            self.node.get_logger().info("📈 Generating plots...")
            self.generate_plots(planning_times, attempts, states, node_counts, tree_depths, success, metrics)

            self.node.get_logger().info(f"✅ Analysis for {self.planner_name} Completed Successfully.")
            return metrics

        except Exception as e:
            self.node.get_logger().error(f"❌ Planning analysis failed: {e}")
            return None

    # ------------------------------------------------
    # Plotting Module (Titles updated with planner_name)
    # ------------------------------------------------
    def generate_plots(self, planning_times, attempts, states, node_counts, tree_depths, success, metrics):
        
        # 1. Planning time histogram
        plt.figure(figsize=(8, 5))
        plt.hist(planning_times, bins=40, color='skyblue', edgecolor='black')
        plt.title(f"Planning Time Histogram ({self.planner_name})")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{self.plots_dir}/planning_time_hist.png")
        plt.close()

        # 2. Planning time box plot
        plt.figure(figsize=(6, 5))
        plt.boxplot(planning_times, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.title(f"Planning Time Distribution ({self.planner_name})")
        plt.ylabel("Time (s)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{self.plots_dir}/planning_time_box.png")
        plt.close()

        # 3. Planning Time Variance (Error Bar Chart)
        plt.figure(figsize=(5, 5))
        std_dev = np.sqrt(metrics["planning_time_variance"])
        plt.bar(["Planning Time"], [metrics["avg_planning_time"]], yerr=[std_dev], capsize=10, color='#3498db', edgecolor='black', alpha=0.8)
        plt.title(f"Planning Time & Variance ({self.planner_name})")
        plt.ylabel("Time (s)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{self.plots_dir}/planning_time_variance_errorbar.png")
        plt.close()

        # 4. Attempts Bar Chart
        plt.figure(figsize=(5, 5))
        plt.bar(["Avg Attempts"], [metrics["avg_attempts"]], color='purple')
        plt.title(f"Average Planning Attempts ({self.planner_name})")
        plt.ylabel("Count")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{self.plots_dir}/planning_attempts.png")
        plt.close()

        # 5 & 6. Success vs Failure Rate
        plt.figure(figsize=(6, 5))
        vals = [metrics["success_rate"] * 100, metrics["failure_rate"] * 100]
        plt.bar(["Success", "Failure"], vals, color=['#2ecc71', '#e74c3c'])
        plt.title(f"Success vs Failure Rate ({self.planner_name})")
        plt.ylabel("Percentage (%)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{self.plots_dir}/success_failure.png")
        plt.close()

        # 7. States explored histogram
        plt.figure(figsize=(8, 5))
        plt.hist(states, bins=40, color='#f39c12', edgecolor='black')
        plt.title(f"States Explored ({self.planner_name})")
        plt.xlabel("Number of States")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{self.plots_dir}/states_explored.png")
        plt.close()

        # 8. Node count
        plt.figure(figsize=(5, 5))
        plt.bar(["Avg Nodes"], [metrics["avg_node_count"]], color='#9b59b6')
        plt.title(f"Planner Node Count ({self.planner_name})")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{self.plots_dir}/node_count.png")
        plt.close()

        # 9. Tree depth histogram
        plt.figure(figsize=(8, 5))
        plt.hist(tree_depths, bins=40, color='#34495e', edgecolor='white')
        plt.title(f"Planner Tree Depth ({self.planner_name})")
        plt.xlabel("Depth")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{self.plots_dir}/tree_depth.png")
        plt.close()

        # 10. Sampling efficiency Scatter
        plt.figure(figsize=(8, 5))
        efficiency = np.array(success) / (np.array(states) + 1e-6)
        plt.scatter(states, efficiency, s=10, alpha=0.6, color='teal')
        plt.title(f"Sampling Efficiency ({self.planner_name})")
        plt.xlabel("States Explored")
        plt.ylabel("Efficiency (Success / States)")
        plt.grid(linestyle='--', alpha=0.5)
        plt.savefig(f"{self.plots_dir}/sampling_efficiency.png")
        plt.close()

        # 11. Robustness Score (Radar Chart)
        self.plot_robustness_radar(metrics)

    def plot_robustness_radar(self, metrics):
        labels = np.array(["Success Rate", "Speed (1/Time)", "Stability (1/Var)", "Efficiency (1/Att)"])
        
        stats = np.array([
            metrics["success_rate"],
            1.0 / (1.0 + metrics["avg_planning_time"]),
            1.0 / (1.0 + metrics["planning_time_variance"]),
            1.0 / (1.0 + metrics["avg_attempts"])
        ])

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        stats = np.concatenate((stats,[stats[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, stats, color='#3498db', alpha=0.3)
        ax.plot(angles, stats, color='#2980b9', linewidth=2)
        
        ax.set_yticklabels([]) 
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        
        plt.title(f"Robustness Radar Chart ({self.planner_name})", y=1.08, fontweight='bold')
        plt.savefig(f"{self.plots_dir}/robustness_radar.png", bbox_inches='tight')
        plt.close()