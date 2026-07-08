<<<<<<< HEAD
import os
import glob
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import cKDTree

class KinematicComparison:
    def __init__(self):
        # Directories
        self.dir_5dof = Path.home() / "metric_ws/1work_paper/5dof_results"
        self.dir_6dof = Path.home() / "metric_ws/1work_paper/6dof_results"
        
        # Core Metrics
        self.metrics = {
            "calc_time": 0.0,
            "ik_timeout": 50,  
            "collisions_ignored": "NO (Collision free safe poses only.)",
        }

        # Data Containers for all 7 Modules
        self.kinematics = {"5dof": {}, "6dof": {}}
        self.advanced_metrics = {"5dof": {}, "6dof": {}}
        self.dexterity_metrics = {"5dof": {}, "6dof": {}}
        self.ik_metrics = {"5dof": {}, "6dof": {}}
        self.plan_metrics = {"5dof": {}, "6dof": {}}
        self.traj_metrics = {"5dof": {}, "6dof": {}}
        
        self.df_5dof = None
        self.df_6dof = None
        self.shared_mask = None

    def _apply_ieee_style(self):
        """Applies IEEE publication-standard formatting to all matplotlib plots."""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'axes.edgecolor': 'black',
            'axes.linewidth': 1.0,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white'
        })

    def _get_latest_files(self, directory):
        """Helper function to find the latest generated files from all modules."""
        if not directory.exists():
            print(f"[WARNING] Directory not found: {directory}")
            return None, None, None, None, None, None, None
            
        csv_f = sorted(glob.glob(str(directory / "*dataset*.csv")), key=os.path.getmtime, reverse=True)
        json_f = sorted(glob.glob(str(directory / "*metadata.json")), key=os.path.getmtime, reverse=True)
        sum_f = sorted(glob.glob(str(directory / "workspace_metrics/workspace_metrics_summary.csv")), key=os.path.getmtime, reverse=True)
        dex_f = sorted(glob.glob(str(directory / "dexterity_metrics/dexterity_metrics_summary.csv")), key=os.path.getmtime, reverse=True)
        ik_f = sorted(glob.glob(str(directory / "ik_metrics/ik_metrics_summary.csv")), key=os.path.getmtime, reverse=True)
        plan_f = sorted(glob.glob(str(directory / "planning_metrics_*/*_summary.csv")), key=os.path.getmtime, reverse=True)
        traj_f = sorted(glob.glob(str(directory / "trajectory_metrics/trajectory_metrics_summary.csv")), key=os.path.getmtime, reverse=True)
        
        return (csv_f[0] if csv_f else None, 
                json_f[0] if json_f else None, 
                sum_f[0] if sum_f else None,
                dex_f[0] if dex_f else None,
                ik_f[0] if ik_f else None,
                plan_f[0] if plan_f else None,
                traj_f[0] if traj_f else None)

    def load_data(self):
        print("[INFO] Aggregating all modular datasets (Modules 1-7)...")
        
        c5, j5, s5, d5, i5, p5, t5 = self._get_latest_files(self.dir_5dof)
        c6, j6, s6, d6, i6, p6, t6 = self._get_latest_files(self.dir_6dof)

        # 1. Load 5-DOF
        if j5 and c5:
            with open(j5, 'r') as f: self.kinematics['5dof'] = json.load(f)
            self.df_5dof = pd.read_csv(c5).dropna(subset=['x', 'y', 'z'])
        if s5: self.advanced_metrics['5dof'] = pd.read_csv(s5).iloc[0].to_dict()
        if d5: self.dexterity_metrics['5dof'] = pd.read_csv(d5).iloc[0].to_dict()
        if i5: self.ik_metrics['5dof'] = pd.read_csv(i5).iloc[0].to_dict()
        if p5: self.plan_metrics['5dof'] = pd.read_csv(p5).iloc[0].to_dict()
        if t5: self.traj_metrics['5dof'] = pd.read_csv(t5).iloc[0].to_dict()

        # 2. Load 6-DOF
        if j6 and c6:
            with open(j6, 'r') as f: self.kinematics['6dof'] = json.load(f)
            self.df_6dof = pd.read_csv(c6).dropna(subset=['x', 'y', 'z'])
        if s6: self.advanced_metrics['6dof'] = pd.read_csv(s6).iloc[0].to_dict()
        if d6: self.dexterity_metrics['6dof'] = pd.read_csv(d6).iloc[0].to_dict()
        if i6: self.ik_metrics['6dof'] = pd.read_csv(i6).iloc[0].to_dict()
        if p6: self.plan_metrics['6dof'] = pd.read_csv(p6).iloc[0].to_dict()
        if t6: self.traj_metrics['6dof'] = pd.read_csv(t6).iloc[0].to_dict()

    def calculate_metrics(self):
        print("[INFO] Computing Final Analytics...")
        start_time = time.time()

        if self.df_5dof is None or self.df_6dof is None:
            print("[ERROR] Missing datasets. Aborting cross-calculations.")
            return

        total_5 = len(self.df_5dof)
        total_6 = len(self.df_6dof)
        self.metrics["total_poses"] = total_5 + total_6
        self.metrics["total_5dof"] = total_5
        self.metrics["total_6dof"] = total_6

        ik_success_rate_6 = self.ik_metrics.get('6dof', {}).get('ik_success_rate', 0.0)
        self.metrics["success_6dof"] = int(total_6 * ik_success_rate_6)
        self.metrics["failed_6dof"] = total_6 - self.metrics["success_6dof"]
        self.metrics["rate_6dof"] = ik_success_rate_6 * 100.0

        pts_5 = self.df_5dof[['x', 'y', 'z']].values
        pts_6 = self.df_6dof[['x', 'y', 'z']].values

        tree_6dof = cKDTree(pts_6)
        distances, _ = tree_6dof.query(pts_5, distance_upper_bound=0.02) 
        
        self.shared_mask = distances != np.inf
        shared_count = np.sum(self.shared_mask)
        unreachable_count = total_5 - shared_count

        self.metrics["shared_5dof_6dof"] = shared_count
        self.metrics["unreachable_by_6dof"] = unreachable_count
        self.metrics["shared_space_pct"] = (shared_count / total_5 * 100) if total_5 > 0 else 0
        self.metrics["missed_space_pct"] = (unreachable_count / total_5 * 100) if total_5 > 0 else 0

        self.metrics["calc_time"] = time.time() - start_time

    def generate_report(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = f"""
📊 KINEMATIC REDUNDANCY & REACHABILITY VERIFICATION REPORT
Timestamp:                 {timestamp}
Total Poses Evaluated:     {self.metrics.get('total_poses', 0)}
Total Computation Time:    {self.metrics['calc_time']:.2f} seconds
IK Solver Timeout:         {self.metrics['ik_timeout']}ms per pose
Collisions Ignored:        {self.metrics['collisions_ignored']}

⚙️  BASELINE: 6-DOF SELF-REPRODUCTION (Solver Reliability)
Total 6-DOF Poses Tested:  {self.metrics.get('total_6dof', 0)}
Successfully Reached:      {self.metrics.get('success_6dof', 0)}
Failed (IK Timeout/Sing):  {self.metrics.get('failed_6dof', 0)}
IK Solver Success Rate:    {self.metrics.get('rate_6dof', 0):.2f}%

🚀 EXPERIMENTAL: 5-DOF TO 6-DOF BACKWARDS COMPATIBILITY
Total 5-DOF Poses Tested:  {self.metrics.get('total_5dof', 0)}
Reached by 6-DOF (Shared): {self.metrics.get('shared_5dof_6dof', 0)}
Unreachable by 6-DOF:      {self.metrics.get('unreachable_by_6dof', 0)}
Shared Kinematic Space:    {self.metrics.get('shared_space_pct', 0):.2f}%
5-DOF Workspace missed by 6-DOF: {self.metrics.get('missed_space_pct', 0):.2f}%

---
📐 ADVANCED DEXTERITY, VOLUMETRICS, & IK PERFORMANCE
                        [5-DOF]                 [6-DOF]
IK Success Rate:        {self.ik_metrics.get('5dof', {}).get('ik_success_rate', 0)*100:.2f}%                 {self.ik_metrics.get('6dof', {}).get('ik_success_rate', 0)*100:.2f}%
Avg IK Solve Time:      {self.ik_metrics.get('5dof', {}).get('ik_mean_time', 0):.4f} sec            {self.ik_metrics.get('6dof', {}).get('ik_mean_time', 0):.4f} sec
Global Dexterity (GDI): {self.dexterity_metrics.get('5dof', {}).get('gdi', 0):.4f}                  {self.dexterity_metrics.get('6dof', {}).get('gdi', 0):.4f}
Singularity Ratio:      {self.dexterity_metrics.get('5dof', {}).get('singularity_ratio', 0)*100:.2f}%                 {self.dexterity_metrics.get('6dof', {}).get('singularity_ratio', 0)*100:.2f}%
Convex Hull Volume:     {self.advanced_metrics.get('5dof', {}).get('workspace_volume', 0):.4f} m³           {self.advanced_metrics.get('6dof', {}).get('workspace_volume', 0):.4f} m³

---
🧠 PATH PLANNING & SEARCH EFFICIENCY (MoveIt)
                        [5-DOF]                 [6-DOF]
Plan Success Rate:      {self.plan_metrics.get('5dof', {}).get('success_rate', 0)*100:.2f}%                 {self.plan_metrics.get('6dof', {}).get('success_rate', 0)*100:.2f}%
Avg Planning Time:      {self.plan_metrics.get('5dof', {}).get('avg_planning_time', 0):.4f} sec            {self.plan_metrics.get('6dof', {}).get('avg_planning_time', 0):.4f} sec
Avg States Explored:    {self.plan_metrics.get('5dof', {}).get('avg_states_explored', 0):.1f}                  {self.plan_metrics.get('6dof', {}).get('avg_states_explored', 0):.1f}
Avg Node Count:         {self.plan_metrics.get('5dof', {}).get('avg_node_count', 0):.1f}                  {self.plan_metrics.get('6dof', {}).get('avg_node_count', 0):.1f}

---
⚡ TRAJECTORY DYNAMICS & MOTION QUALITY
                        [5-DOF]                 [6-DOF]
Path Efficiency (0-1):  {self.traj_metrics.get('5dof', {}).get('avg_path_efficiency', 0):.4f}                  {self.traj_metrics.get('6dof', {}).get('avg_path_efficiency', 0):.4f}
Joint Path Length:      {self.traj_metrics.get('5dof', {}).get('avg_joint_path_length', 0):.2f} rad               {self.traj_metrics.get('6dof', {}).get('avg_joint_path_length', 0):.2f} rad
Motion Energy:          {self.traj_metrics.get('5dof', {}).get('avg_motion_energy', 0):.2f} J                 {self.traj_metrics.get('6dof', {}).get('avg_motion_energy', 0):.2f} J
Smoothness (Jerk²):     {self.traj_metrics.get('5dof', {}).get('avg_smoothness', 0):.2f}                 {self.traj_metrics.get('6dof', {}).get('avg_smoothness', 0):.2f}
Exec Duration:          {self.traj_metrics.get('5dof', {}).get('avg_trajectory_duration', 0):.2f} sec               {self.traj_metrics.get('6dof', {}).get('avg_trajectory_duration', 0):.2f} sec
"""
        print(report)
        with open("kinematic_redundancy_report.txt", "w") as f:
            f.write(report)
        print("[INFO] Report saved to kinematic_redundancy_report.txt")

    # =========================================================
    # IEEE STANDARD PLOTTING
    # =========================================================
    def plot_comparisons(self):
        print("[INFO] Generating IEEE standard visual comparisons...")
        self._apply_ieee_style() 

        if self.df_5dof is not None and self.df_6dof is not None:
            self._plot_unreachable_workspace()
        
        if self.dexterity_metrics.get('5dof') and self.dexterity_metrics.get('6dof'):
            self._plot_dexterity_barchart()
            
        if self.ik_metrics.get('5dof') and self.ik_metrics.get('6dof'):
            self._plot_ik_performance_barchart()
            
        if self.plan_metrics.get('5dof') and self.plan_metrics.get('6dof'):
            self._plot_planning_performance_barchart()
            self._plot_planning_search_efficiency()
            
        if self.traj_metrics.get('5dof') and self.traj_metrics.get('6dof'):
            self._plot_trajectory_quality_barchart()
            self._plot_trajectory_dynamics_barchart()

    def _plot_unreachable_workspace(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        pts_5 = self.df_5dof[['x', 'y', 'z']].values
        shared_pts = pts_5[self.shared_mask]
        unreachable_pts = pts_5[~self.shared_mask]

        np.random.seed(42)
        idx_s = np.random.choice(len(shared_pts), min(len(shared_pts), 4000), replace=False) if len(shared_pts) > 0 else []
        idx_u = np.random.choice(len(unreachable_pts), min(len(unreachable_pts), 4000), replace=False) if len(unreachable_pts) > 0 else []

        if len(idx_s) > 0:
            ax.scatter(shared_pts[idx_s, 0], shared_pts[idx_s, 1], shared_pts[idx_s, 2], 
                       c='#1f77b4', s=2, alpha=0.3, label=f'Shared Space ({self.metrics["shared_space_pct"]:.1f}%)')
        
        if len(idx_u) > 0:
            ax.scatter(unreachable_pts[idx_u, 0], unreachable_pts[idx_u, 1], unreachable_pts[idx_u, 2], 
                       c='#d62728', s=6, alpha=0.9, marker='x', label=f'Unreachable by 6-DOF ({self.metrics["missed_space_pct"]:.1f}%)')

        ax.set_xlabel('X Axis (m)')
        ax.set_ylabel('Y Axis (m)')
        ax.set_zlabel('Z Axis (m)')
        ax.set_title('Volumetric Compatibility: 5-DOF Workspace Overlap')
        
        leg = ax.legend(loc='upper right', framealpha=1.0)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
            lh._sizes = [50]

        plt.tight_layout()
        plt.savefig("ieee_workspace_compatibility_3d.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_dexterity_barchart(self):
        labels = ['Manipulability (x10)', 'Global Dexterity', 'Singularity %']
        dm5 = self.dexterity_metrics.get('5dof', {})
        dm6 = self.dexterity_metrics.get('6dof', {})

        vals_5 = [dm5.get('manipulability_mean', 0)*10, dm5.get('gdi', 0), dm5.get('singularity_ratio', 0)*100]
        vals_6 = [dm6.get('manipulability_mean', 0)*10, dm6.get('gdi', 0), dm6.get('singularity_ratio', 0)*100]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 4.5))
        rects1 = ax.bar(x - width/2, vals_5, width, label='5-DOF', color='#1f77b4', edgecolor='black', hatch='//')
        rects2 = ax.bar(x + width/2, vals_6, width, label='6-DOF', color='#ff7f0e', edgecolor='black', hatch='\\\\')

        ax.set_ylabel('Score / Percentage')
        ax.set_title('Kinematic Dexterity and Singularity Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig("ieee_dexterity_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_ik_performance_barchart(self):
        ik5 = self.ik_metrics.get('5dof', {})
        ik6 = self.ik_metrics.get('6dof', {})

        success_vals = [ik5.get('ik_success_rate', 0)*100, ik6.get('ik_success_rate', 0)*100]
        time_vals = [ik5.get('ik_mean_time', 0)*1000, ik6.get('ik_mean_time', 0)*1000]

        x = np.arange(2)
        width = 0.3

        fig, ax1 = plt.subplots(figsize=(6, 4.5))

        ax1.set_ylabel('IK Success Rate (%)', color='black')
        rects1 = ax1.bar(x - width/2, success_vals, width, label='Success Rate (%)', color='#2ca02c', edgecolor='black', hatch='xx')
        ax1.set_ylim(0, 110)

        ax2 = ax1.twinx()  
        ax2.set_ylabel('Mean Solve Time (ms)', color='black')
        rects2 = ax2.bar(x + width/2, time_vals, width, label='Solve Time (ms)', color='#d62728', edgecolor='black', hatch='..')
        ax2.set_ylim(0, max(max(time_vals) * 1.3, 10))

        ax1.set_xticks(x)
        ax1.set_xticklabels(['5-DOF Setup', '6-DOF Setup'])
        plt.title('Inverse Kinematics Performance Comparison')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig("ieee_ik_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_planning_performance_barchart(self):
        p5 = self.plan_metrics.get('5dof', {})
        p6 = self.plan_metrics.get('6dof', {})

        success_vals = [p5.get('success_rate', 0)*100, p6.get('success_rate', 0)*100]
        time_vals = [p5.get('avg_planning_time', 0), p6.get('avg_planning_time', 0)] 

        x = np.arange(2)
        width = 0.3

        fig, ax1 = plt.subplots(figsize=(6, 4.5))
        ax1.set_ylabel('Plan Success Rate (%)', color='black')
        ax1.bar(x - width/2, success_vals, width, label='Success Rate (%)', color='#8c564b', edgecolor='black', hatch='//')
        ax1.set_ylim(0, 110)

        ax2 = ax1.twinx()  
        ax2.set_ylabel('Mean Plan Time (s)', color='black')
        ax2.bar(x + width/2, time_vals, width, label='Plan Time (s)', color='#e377c2', edgecolor='black', hatch='**')
        ax2.set_ylim(0, max(max(time_vals) * 1.3, 1.0))

        ax1.set_xticks(x)
        ax1.set_xticklabels(['5-DOF Setup', '6-DOF Setup'])
        plt.title('Motion Planning Robustness Comparison')
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig("ieee_planning_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_planning_search_efficiency(self):
        p5 = self.plan_metrics.get('5dof', {})
        p6 = self.plan_metrics.get('6dof', {})

        labels = ['States Explored', 'Node Count', 'Tree Depth']
        vals_5 = [p5.get('avg_states_explored', 0), p5.get('avg_node_count', 0), p5.get('avg_tree_depth', 0)]
        vals_6 = [p6.get('avg_states_explored', 0), p6.get('avg_node_count', 0), p6.get('avg_tree_depth', 0)]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(x - width/2, vals_5, width, label='5-DOF', color='#17becf', edgecolor='black', hatch='oo')
        ax.bar(x + width/2, vals_6, width, label='6-DOF', color='#7f7f7f', edgecolor='black', hatch='++')

        ax.set_ylabel('Count (Log Scale)')
        ax.set_title('Path Planner Search Complexity')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig("ieee_planning_search_efficiency.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_trajectory_quality_barchart(self):
        t5 = self.traj_metrics.get('5dof', {})
        t6 = self.traj_metrics.get('6dof', {})

        labels = ['Path Efficiency (0-1)', 'Time Optimality (0-1)']
        vals_5 = [t5.get('avg_path_efficiency', 0), t5.get('avg_time_optimality', 0)]
        vals_6 = [t6.get('avg_path_efficiency', 0), t6.get('avg_time_optimality', 0)]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(6, 4.5))
        rects1 = ax.bar(x - width/2, vals_5, width, label='5-DOF', color='#2ca02c', edgecolor='black', hatch='--')
        rects2 = ax.bar(x + width/2, vals_6, width, label='6-DOF', color='#9467bd', edgecolor='black', hatch='xx')

        ax.set_ylabel('Efficiency Ratio (Higher is better)')
        ax.set_title('Trajectory Efficiency and Optimality')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.2)
        ax.legend()

        for rect in rects1 + rects2:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig("ieee_trajectory_efficiency.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_trajectory_dynamics_barchart(self):
        t5 = self.traj_metrics.get('5dof', {})
        t6 = self.traj_metrics.get('6dof', {})

        labels = ['Motion Energy (J)', 'Smoothness (Jerk²)', 'Motion Cost']
        vals_5 = [t5.get('avg_motion_energy', 0), t5.get('avg_smoothness', 0), t5.get('avg_motion_cost', 0)]
        vals_6 = [t6.get('avg_motion_energy', 0), t6.get('avg_smoothness', 0), t6.get('avg_motion_cost', 0)]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 4.5))
        rects1 = ax.bar(x - width/2, vals_5, width, label='5-DOF', color='#d62728', edgecolor='black', hatch='||')
        rects2 = ax.bar(x + width/2, vals_6, width, label='6-DOF', color='#8c564b', edgecolor='black', hatch='..')

        ax.set_ylabel('Dynamic Penalty (Log Scale)')
        ax.set_title('Dynamic Trajectory Penalties (Lower is better)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig("ieee_trajectory_dynamics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def run_pipeline(self):
        self.load_data()
        self.calculate_metrics()
        self.generate_report()
        self.plot_comparisons()
        print("\n✅ Kinematic comparison pipeline complete!")
        print("✅ IEEE-standard graphs saved as high-resolution PNGs.")

if __name__ == "__main__":
    evaluator = KinematicComparison()
    evaluator.run_pipeline()
=======
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_label_data(db_path, robot_label):
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM runs", conn)
        df['Robot_Type'] = robot_label
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading {db_path}: {e}")
        return pd.DataFrame()

print("Loading databases...")
df_4dof = load_and_label_data('4dof.db', '4-DOF Arm')
df_5dof = load_and_label_data('5dof.db', '5-DOF Arm')
df_all = pd.concat([df_4dof, df_5dof], ignore_index=True)

print("Cleaning data...")
# Translate the words 'true' and 'false' into 1 and 0
df_all = df_all.replace({'true': 1, 'false': 0, 'True': 1, 'False': 0})

# ADDED NEW METRICS TO THE CLEANING LIST
metrics_to_convert =[
    'solved', 'time', 'path_plan_length', 'path_plan_smoothness', 
    'final_path_length', 'process_time', 'average_waypoint_distance'
]

for metric in metrics_to_convert:
    if metric in df_all.columns:
        df_all[metric] = pd.to_numeric(df_all[metric], errors='coerce')

df_all['solved'] = df_all['solved'].fillna(0)
df_successful = df_all[df_all['solved'] == 1].copy()

print("Generating 6 graphs...")

# Set up a 2x3 grid (18 units wide by 10 units tall)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
colors =["#FF6B6B", "#4ECDC4"] 
sns.set_palette(sns.color_palette(colors))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Comprehensive Performance: 4-DOF vs 5-DOF Manipulator (300 Poses)", fontsize=20, fontweight='bold')

# --- Plot 1: Success Rate ---
success_rates = df_all.groupby('Robot_Type')['solved'].mean() * 100
success_rates.plot(kind='bar', ax=axes[0, 0], color=colors, edgecolor='black')
axes[0, 0].set_title('Success Rate', fontweight='bold')
axes[0, 0].set_ylabel('Success (%)')
axes[0, 0].set_ylim(0, 105)
axes[0, 0].tick_params(axis='x', rotation=0)

# --- Plot 2: Planning Time ---
sns.boxplot(data=df_successful, x='Robot_Type', y='time', ax=axes[0, 1], width=0.5)
axes[0, 1].set_title('Raw Planning Time', fontweight='bold')
axes[0, 1].set_ylabel('Time (sec)')
axes[0, 1].set_xlabel('')

# --- Plot 3: Raw Path Length ---
sns.boxplot(data=df_successful, x='Robot_Type', y='path_plan_length', ax=axes[0, 2], width=0.5)
axes[0, 2].set_title('Raw Path Length (Before Smoothing)', fontweight='bold')
axes[0, 2].set_ylabel('Length (rad/m)')
axes[0, 2].set_xlabel('')

# --- Plot 4: Final Path Length (Smoothed) ---
if 'final_path_length' in df_successful.columns:
    sns.boxplot(data=df_successful, x='Robot_Type', y='final_path_length', ax=axes[1, 0], width=0.5)
    axes[1, 0].set_title('Final Path Length (After Smoothing)', fontweight='bold')
    axes[1, 0].set_ylabel('Length (rad/m)')
    axes[1, 0].set_xlabel('')

# --- Plot 5: Average Waypoint Distance ---
if 'average_waypoint_distance' in df_successful.columns:
    sns.boxplot(data=df_successful, x='Robot_Type', y='average_waypoint_distance', ax=axes[1, 1], width=0.5)
    axes[1, 1].set_title('Average Waypoint Distance', fontweight='bold')
    axes[1, 1].set_ylabel('Distance')
    axes[1, 1].set_xlabel('')

# --- Plot 6: Total Process Time ---
if 'process_time' in df_successful.columns:
    sns.boxplot(data=df_successful, x='Robot_Type', y='process_time', ax=axes[1, 2], width=0.5)
    axes[1, 2].set_title('Total Process Time (Plan + Smooth)', fontweight='bold')
    axes[1, 2].set_ylabel('Time (sec)')
    axes[1, 2].set_xlabel('')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Robot_Comparison_6_Metrics.pdf", dpi=300, bbox_inches='tight')
plt.show()



# import sqlite3
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ==========================================
# # 1. FUNCTION TO LOAD AND PREPARE DATA
# # ==========================================
# def load_and_label_data(db_path, robot_label):
#     """Connects to the SQLite DB, extracts the runs, and adds a label."""
#     try:
#         conn = sqlite3.connect(db_path)
#         # The benchmark metrics are stored in the 'runs' table
#         df = pd.read_sql_query("SELECT * FROM runs", conn)
#         df['Robot_Type'] = robot_label
#         conn.close()
#         return df
#     except Exception as e:
#         print(f"Error loading {db_path}: {e}")
#         return pd.DataFrame()

# # Load both databases
# print("Loading databases...")
# df_4dof = load_and_label_data('4dof.db', '4-DOF Arm')
# df_5dof = load_and_label_data('5dof.db', '5-DOF Arm')

# # Combine them into a single Master DataFrame
# df_all = pd.concat([df_4dof, df_5dof], ignore_index=True)

# # ==========================================
# # 2. DATA CLEANING
# # ==========================================
# # print("Cleaning data...")
# # # Convert numerical columns to actual floats (sometimes SQLite stores them as strings)
# # metrics_to_convert =['solved', 'time', 'path_plan_length', 'path_plan_smoothness']
# # for metric in metrics_to_convert:
# #     if metric in df_all.columns:
# #         df_all[metric] = pd.to_numeric(df_all[metric], errors='coerce')

# # # Fill missing 'solved' values with 0 (Failure)
# # df_all['solved'] = df_all['solved'].fillna(0)

# # # CREATE A DATAFRAME FOR SUCCESSFUL RUNS ONLY
# # # We do this because comparing the "planning time" or "path length" of a failed run 
# # # is scientifically inaccurate for a research paper.
# # df_successful = df_all[df_all['solved'] == 1].copy()

# # ==========================================
# # 2. DATA CLEANING (UPDATED FIX)
# # ==========================================
# print("Cleaning data...")

# # FIX: Translate the words 'true' and 'false' into 1 and 0
# df_all = df_all.replace({'true': 1, 'false': 0, 'True': 1, 'False': 0})

# metrics_to_convert =['solved', 'time', 'path_plan_length', 'path_plan_smoothness']
# for metric in metrics_to_convert:
#     if metric in df_all.columns:
#         df_all[metric] = pd.to_numeric(df_all[metric], errors='coerce')

# df_all['solved'] = df_all['solved'].fillna(0)
# df_successful = df_all[df_all['solved'] == 1].copy()

# # ==========================================
# # 3. PLOTTING PUBLICATION-QUALITY GRAPHS
# # ==========================================
# print("Generating graphs...")

# # Set the visual style to look highly professional (ideal for IEEE/Springer papers)
# sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
# colors = ["#FF6B6B", "#4ECDC4"] # Distinct, colorblind-friendly colors
# sns.set_palette(sns.color_palette(colors))

# # Create a figure with 4 subplots (2x2 grid)
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# fig.suptitle("Performance Comparison: 4-DOF vs 5-DOF Manipulator (300 Poses)", fontsize=18, fontweight='bold')

# # --- Plot 1: Success Rate (Bar Chart) ---
# # Calculates the percentage of times 'solved' == 1
# success_rates = df_all.groupby('Robot_Type')['solved'].mean() * 100
# success_rates.plot(kind='bar', ax=axes[0, 0], color=colors, edgecolor='black')
# axes[0, 0].set_title('Pose Solving Success Rate', fontweight='bold')
# axes[0, 0].set_ylabel('Success Rate (%)')
# axes[0, 0].set_ylim(0, 105)
# axes[0, 0].tick_params(axis='x', rotation=0)

# # --- Plot 2: Planning Time (Box Plot) ---
# # Shows the distribution of time taken to find a solution (lower is better)
# sns.boxplot(data=df_successful, x='Robot_Type', y='time', ax=axes[0, 1], width=0.5)
# axes[0, 1].set_title('Planning Time (Successful Runs Only)', fontweight='bold')
# axes[0, 1].set_ylabel('Time (seconds)')
# axes[0, 1].set_xlabel('')

# # --- Plot 3: Path Length (Box Plot) ---
# # Shows how efficient the movements are (lower is better)
# sns.boxplot(data=df_successful, x='Robot_Type', y='path_plan_length', ax=axes[1, 0], width=0.5)
# axes[1, 0].set_title('Generated Path Length', fontweight='bold')
# axes[1, 0].set_ylabel('Path Length (radians/meters)')
# axes[1, 0].set_xlabel('')

# # --- Plot 4: Path Smoothness (Box Plot) ---
# # Shows how smooth the trajectory is (lower is generally better, indicating fewer sharp turns)
# if 'path_plan_smoothness' in df_successful.columns:
#     sns.boxplot(data=df_successful, x='Robot_Type', y='path_plan_smoothness', ax=axes[1, 1], width=0.5)
#     axes[1, 1].set_title('Path Smoothness', fontweight='bold')
#     axes[1, 1].set_ylabel('Smoothness Score')
#     axes[1, 1].set_xlabel('')
# else:
#     axes[1, 1].text(0.5, 0.5, 'Smoothness metric not found', ha='center', va='center')

# # Adjust layout to prevent text from overlapping
# plt.tight_layout(rect=[0, 0, 1, 0.96])

# # ==========================================
# # 4. SAVE AND DISPLAY
# # ==========================================
# # Save as a high-resolution PDF and PNG for your LaTeX/Word document
# plt.savefig("Robot_Comparison_Metrics.pdf", dpi=300, bbox_inches='tight')
# plt.savefig("Robot_Comparison_Metrics.png", dpi=300, bbox_inches='tight')
# print("Saved graphs as 'Robot_Comparison_Metrics.pdf' and '.png'")

# # Show the graphs on your screen
# plt.show()

# # ==========================================
# # 5. PRINT STATISTICAL SUMMARY FOR PAPER TEXT
# # ==========================================
# print("\n" + "="*50)
# print("STATISTICAL SUMMARY FOR YOUR RESEARCH PAPER:")
# print("="*50)
# print(f"Total Poses Evaluated: {len(df_4dof)} per robot")
# print(f"\nSuccess Rates:")
# print(f"4-DOF: {success_rates['4-DOF Arm']:.1f}%")
# print(f"5-DOF: {success_rates['5-DOF Arm']:.1f}%")

# print(f"\nAverage Planning Time (Successful Runs):")
# print(f"4-DOF: {df_successful[df_successful['Robot_Type']=='4-DOF Arm']['time'].mean():.4f} seconds")
# print(f"5-DOF: {df_successful[df_successful['Robot_Type']=='5-DOF Arm']['time'].mean():.4f} seconds")

# print(f"\nAverage Path Length (Successful Runs):")
# print(f"4-DOF: {df_successful[df_successful['Robot_Type']=='4-DOF Arm']['path_plan_length'].mean():.4f}")
# print(f"5-DOF: {df_successful[df_successful['Robot_Type']=='5-DOF Arm']['path_plan_length'].mean():.4f}")
>>>>>>> 1abd165d004be5ab2a0bac131630d4a2d1f63bf5
