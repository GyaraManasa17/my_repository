#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK, GetStateValidity
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
import numpy as np
import pandas as pd
import math
import time
from datetime import datetime
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from scipy.stats import qmc  # ✅ ADDED: Low-Discrepancy Sampling

class AdvancedWorkspaceAnalyzer(Node):
    def __init__(self):
        super().__init__("advanced_workspace_analyzer")
        
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        self.state_validity_client = self.create_client(GetStateValidity, "/check_state_validity")
        
        self.joints =["joint1", "joint2", "joint3", "joint4", "joint5_roll"]
        self.end_link = "end_effector_link"
        self.limits =[
            (-math.pi, math.pi),  
            (-1.5, 1.5),          
            (-1.5, 1.4),          
            (-1.7, 1.97),         
            (-3.14, 3.14)         
        ]
        
    def wait_for_services(self):
        self.get_logger().info("Waiting for MoveIt services...")
        fk_ready = self.fk_client.wait_for_service(timeout_sec=5.0)
        sv_ready = self.state_validity_client.wait_for_service(timeout_sec=5.0)
        return fk_ready and sv_ready

    def run_advanced_analysis(self, num_samples=8000):
        if not self.wait_for_services():
            self.get_logger().error("Services not available!")
            return
            
        print("\n" + "="*80)
        print(f"🚀 STARTING HIGH-PRECISION WORKSPACE ANALYSIS ({num_samples} samples)")
        print("="*80)
        
        # ✅ FIX: Halton Low-Discrepancy Sequence instead of purely random
        print("🔢 Generating Halton Sequence (Low-Discrepancy Sampling)...")
        sampler = qmc.Halton(d=5, scramble=True)
        samples = sampler.random(n=num_samples)
        
        for i in range(5):
            min_val, max_val = self.limits[i]
            samples[:, i] = samples[:, i] * (max_val - min_val) + min_val
            
        results =[]
        
        req = GetPositionFK.Request()
        req.header.frame_id = "link1" 
        req.fk_link_names =[self.end_link]
        
        sv_req = GetStateValidity.Request()
        sv_req.group_name = "arm"
        
        robot_state = RobotState()
        joint_state = JointState(name=self.joints)
        
        start_time = time.time()
        
        for i in range(num_samples):
            if i % 1000 == 0 and i > 0:
                rate = i / (time.time() - start_time)
                print(f"⏳ Processed {i}/{num_samples} poses... ({rate:.1f} poses/sec)")
            
            joint_state.position = [float(v) for v in samples[i]]
            robot_state.joint_state = joint_state
            
            sv_req.robot_state = robot_state
            sv_future = self.state_validity_client.call_async(sv_req)
            
            # ✅ FIX: Added Timeout to prevent freezing
            rclpy.spin_until_future_complete(self, sv_future, timeout_sec=2.0)
            if not sv_future.done():
                self.get_logger().warn("State Validity timeout, skipping...")
                continue
                
            sv_resp = sv_future.result()

            if not sv_resp or not sv_resp.valid:
                continue
            
            req.robot_state = robot_state
            future = self.fk_client.call_async(req)
            
            # ✅ FIX: Added Timeout for FK
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if not future.done():
                self.get_logger().warn("FK timeout, skipping...")
                continue
            
            resp = future.result()
            if resp and resp.error_code.val == resp.error_code.SUCCESS:
                pose = resp.pose_stamped[0].pose
                x, y, z = pose.position.x, pose.position.y, pose.position.z
                
                # ✅ FIX: Base Safety Filter (CRITICAL for Script 2)
                if z < 0.05:
                    continue
                
                q = pose.orientation
                rot = R.from_quat([q.x, q.y, q.z, q.w])
                roll, pitch, yaw = rot.as_euler('xyz', degrees=False)
                
                results.append([
                    samples[i][0], samples[i][1], samples[i][2], samples[i][3], samples[i][4],
                    x, y, z, roll, pitch, yaw, q.x, q.y, q.z, q.w
                ])
        
        total_time = time.time() - start_time
        print(f"✅ Extracted {len(results)} valid Cartesian points in {total_time:.2f} seconds.")
        
        self.process_advanced_results(results, num_samples)

    def process_advanced_results(self, results, num_samples):
        if not results:
            print("❌ No valid FK poses returned.")
            return
            
        data = np.array(results)
        points_3d = data[:, 5:8]
        valid_poses_count = len(results)

        # ✅ FIX: Calculate Workspace Metrics for Paper
        try:
            hull = ConvexHull(points_3d)
            volume = hull.volume
        except Exception:
            volume = 0.0

        print("\n" + "="*50)
        print(" 📊 WORKSPACE METRICS (FOR PAPER)")
        print("="*50)
        print(f"📦 Workspace Volume: {volume:.6f} m³")
        print(f"📈 Valid Pose Ratio: {valid_poses_count / num_samples:.3f} ({valid_poses_count}/{num_samples})")
        print(f"⬆️ Z Range:          {points_3d[:,2].min():.3f} m  to  {points_3d[:,2].max():.3f} m")
        print("="*50 + "\n")

        columns =['j1', 'j2', 'j3', 'j4', 'j5', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'qx', 'qy', 'qz', 'qw']
        poses_df = pd.DataFrame(data[:, :15], columns=columns)
        
        # ✅ FIX: Save to Safe Base file for Script 2
        poses_file = "omx_safe_base_poses.csv"
        poses_df.to_csv(poses_file, index=False)
        print(f"💾 Saved {valid_poses_count} SAFE Base poses to: {poses_file}")

def main(args=None):
    rclpy.init(args=args)
    analyzer = AdvancedWorkspaceAnalyzer()
    analyzer.run_advanced_analysis(num_samples=8000)
    analyzer.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()