#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import pandas as pd
import time
# ==========================================
# ✅ NEW IMPORTS FOR RESEARCH-GRADE OUTPUTS
# ==========================================
import matplotlib.pyplot as plt
from datetime import datetime
# ==========================================

class ReachabilityVerifier(Node):
    def __init__(self):
        super().__init__("reachability_verifier")
        
        # Create IK Client
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.group_name = "arm" 
        self.base_frame = "link1" # Double check this is your 5-DOF root link!
        self.end_link = "end_effector_link"

    def wait_for_service(self):
        self.get_logger().info("Waiting for MoveIt /compute_ik service...")
        return self.ik_client.wait_for_service(timeout_sec=5.0)

    def verify_poses(self, csv_5dof, csv_6dof):
        if not self.wait_for_service():
            self.get_logger().error("Service not available! Is your 5-DOF MoveIt running?")
            return

        print("\n" + "="*80)
        print(" 🔍 STARTING IK VERIFICATION (STATISTICS ONLY)")
        print("="*80)

        # 1. Load Data & Add Source Tracking
        df5 = pd.read_csv(csv_5dof)
        df5['source'] = '5dof'
        print(f"📥 Loaded {len(df5)} poses from the 5-DOF dataset.")
        
        df6 = pd.read_csv(csv_6dof)
        df6['source'] = '6dof'
        print(f"📥 Loaded {len(df6)} poses from the 6-DOF dataset.")
        
        # Combine
        df = pd.concat([df5, df6], ignore_index=True)
        df['reachable_by_6dof'] = False # Default to False
        
        total_poses = len(df)
        print(f"🔢 Total combined poses stored in memory for testing: {total_poses}")
        print("-" * 80)
        
        # Prepare IK Request
        req = GetPositionIK.Request()
        ik_req = PositionIKRequest()
        ik_req.group_name = self.group_name
        ik_req.ik_link_name = self.end_link
        ik_req.avoid_collisions = False 
        ik_req.timeout.sec = 0
        ik_req.timeout.nanosec = 50000000 # 50ms timeout

        start_time = time.time()

        # 2. Query MoveIt IK
        for index, row in df.iterrows():
            if index % 1000 == 0 and index > 0:
                rate = index / (time.time() - start_time)
                print(f"⏳ Checked {index}/{total_poses} poses... ({rate:.1f} poses/sec)")

            # RPY to Quaternion
            rot = R.from_euler('xyz', [row['roll'], row['pitch'], row['yaw']], degrees=False)
            qx, qy, qz, qw = rot.as_quat()

            # Construct Pose
            pose = PoseStamped()
            pose.header.frame_id = self.base_frame
            pose.pose.position.x = row['x']
            pose.pose.position.y = row['y']
            pose.pose.position.z = row['z']
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            ik_req.pose_stamped = pose
            req.ik_request = ik_req

            # Call Service
            future = self.ik_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()
            
            # Error Code 1 == SUCCESS
            if resp and resp.error_code.val == 1:
                df.at[index, 'reachable_by_6dof'] = True

        total_time = time.time() - start_time
        
        # =========================================================================
        # ✅ ONLY CHANGES BELOW THIS LINE: RESEARCH METRICS, TEXT EXPORT, AND PLOTS
        # =========================================================================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        df5_subset = df[df['source'] == '5dof']
        df6_subset = df[df['source'] == '6dof']
        
        total_6dof = len(df6_subset)
        success_6dof = df6_subset['reachable_by_6dof'].sum()
        fail_6dof = total_6dof - success_6dof
        
        total_5dof = len(df5_subset)
        success_5dof = df5_subset['reachable_by_6dof'].sum()
        fail_5dof = total_5dof - success_5dof

        # Advanced Academic Metrics
        ik_reliability = (success_6dof / total_6dof * 100) if total_6dof > 0 else 0
        shared_dexterity = (success_5dof / total_5dof * 100) if total_5dof > 0 else 0
        missed_workspace = 100 - shared_dexterity

        summary_text = (
            f"============================================================\n"
            f" 📊 KINEMATIC REDUNDANCY & REACHABILITY VERIFICATION REPORT\n"
            f"============================================================\n"
            f"Timestamp:                 {timestamp}\n"
            f"Total Poses Evaluated:     {total_poses}\n"
            f"Total Computation Time:    {total_time:.2f} seconds\n"
            f"IK Solver Timeout:         50ms per pose\n"
            f"Collisions Ignored:        Yes (Pure Kinematic Assessment)\n"
            f"------------------------------------------------------------\n"
            f" ⚙️  BASELINE: 6-DOF SELF-REPRODUCTION (Solver Reliability)\n"
            f"------------------------------------------------------------\n"
            f"Total 6-DOF Poses Tested:  {total_6dof}\n"
            f"Successfully Reached:      {success_6dof}\n"
            f"Failed (Singularities):    {fail_6dof}\n"
            f"IK Solver Success Rate:    {ik_reliability:.2f}%\n"
            f"------------------------------------------------------------\n"
            f" 🚀 EXPERIMENTAL: 5-DOF TO 6-DOF BACKWARDS COMPATIBILITY\n"
            f"------------------------------------------------------------\n"
            f"Total 5-DOF Poses Tested:  {total_5dof}\n"
            f"Reached by 6-DOF (Shared): {success_5dof}\n"
            f"Unreachable by 6-DOF:      {fail_5dof}\n"
            f"Shared Kinematic Space:    {shared_dexterity:.2f}%\n"
            f"5-DOF Workspace missed by 6-DOF: {missed_workspace:.2f}%\n"
            f"============================================================\n"
        )
        
        # Print and Save Text Report
        print(summary_text)
        report_filename = f"reachability_report_6dof_target_{timestamp}.txt"
        with open(report_filename, "w") as f:
            f.write(summary_text)

        # Generate Publication-Ready Plot
        print("🎨 Generating High-Resolution 3D Workspace and Statistic Plots...")
        fig = plt.figure(figsize=(16, 7))
        
        # Subplot 1: Bar Chart of Statistics
        ax1 = fig.add_subplot(121)
        categories =['6-DOF Self-Test', '5-DOF Overlap Test']
        success_rates = [ik_reliability, shared_dexterity]
        bars = ax1.bar(categories, success_rates, color=['#9C27B0', '#FF9800'])
        ax1.set_ylim(0, 110)
        ax1.set_ylabel('IK Success Rate / Reachability (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Kinematic Reachability Comparison (Target: 6-DOF)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%', 
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

        # Subplot 2: 3D Mapping of 5-DOF Poses (Showing Overlap Compatibility)
        ax2 = fig.add_subplot(122, projection='3d')
        # Filter only 5DOF poses
        df_5_reachable = df5_subset[df5_subset['reachable_by_6dof'] == True]
        df_5_unreachable = df5_subset[df5_subset['reachable_by_6dof'] == False]
        
        ax2.scatter(df_5_reachable['x'], df_5_reachable['y'], df_5_reachable['z'], 
                    c='green', alpha=0.3, s=5, label='Shared Workspace (Reached by 6-DOF)')
        ax2.scatter(df_5_unreachable['x'], df_5_unreachable['y'], df_5_unreachable['z'], 
                    c='red', alpha=0.3, s=5, label='Exclusive to 5-DOF (Unreachable by 6-DOF)')
        
        ax2.set_xlabel('X (m)', fontweight='bold')
        ax2.set_ylabel('Y (m)', fontweight='bold')
        ax2.set_zlabel('Z (m)', fontweight='bold')
        ax2.set_title('5-DOF to 6-DOF Backwards Compatibility Map', fontsize=14, fontweight='bold')
        
        # Make legend clear and highly visible
        leg = ax2.legend(loc="upper right", markerscale=5)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
            
        plt.tight_layout()
        plot_filename = f"reachability_analysis_6dof_target_plot_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300)
        
        # Save Dataset
        out_filename = f"verified_kinematic_overlap_6dof_{timestamp}.csv"
        df.to_csv(out_filename, index=False)
        print(f"💾 Saved Dataset to:       {out_filename}")
        print(f"📄 Saved Text Report to:   {report_filename}")
        print(f"📈 Saved Analysis Plot to: {plot_filename}\n")
        # =========================================================================

def main(args=None):
    rclpy.init(args=args)
    
    # === UPDATE THESE FILENAMES TO MATCH YOUR SAVED CSVs ===
    csv_5dof = "omx_all_reachable_poses_5dof.csv"
    csv_6dof = "omx_all_reachable_poses_6dof.csv"
    
    verifier = ReachabilityVerifier()
    verifier.verify_poses(csv_5dof, csv_6dof)
    
    verifier.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()