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
import alphashape 

class AdvancedWorkspaceAnalyzer(Node):
    def __init__(self):
        super().__init__("advanced_workspace_analyzer")
        
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        self.state_validity_client = self.create_client(GetStateValidity, "/check_state_validity")
        
        # OpenManipulator-X config
        self.joints =["joint1", "joint2", "joint3", "joint4", "joint5_roll"]
        self.end_link = "end_effector_link"
        self.limits =[
            (-math.pi, math.pi),  # joint1
            (-1.5, 1.5),          # joint2
            (-1.5, 1.4),          # joint3
            (-1.7, 1.97),         # joint4
            (-3.14, 3.14)         # joint5
        ]
        
    def wait_for_services(self):
        self.get_logger().info("Waiting for MoveIt services...")
        fk_ready = self.fk_client.wait_for_service(timeout_sec=5.0)
        sv_ready = self.state_validity_client.wait_for_service(timeout_sec=5.0)
        return fk_ready and sv_ready

    def run_advanced_analysis(self, num_samples=8000):
        if not self.wait_for_services():
            self.get_logger().error("Services not available! Is MoveIt running?")
            return
            
        print("\n" + "="*80)
        print(f"🚀 STARTING HIGH-PRECISION WORKSPACE ANALYSIS ({num_samples} samples)")
        print("="*80)
        
        samples = np.zeros((num_samples, 5))
        for i in range(5):
            min_val, max_val = self.limits[i]
            samples[:, i] = np.random.uniform(min_val, max_val, num_samples)
            
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
            rclpy.spin_until_future_complete(self, sv_future)
            sv_resp = sv_future.result()

            if not sv_resp or not sv_resp.valid:
                continue
            
            req.robot_state = robot_state
            future = self.fk_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            
            resp = future.result()
            if resp and resp.error_code.val == resp.error_code.SUCCESS:
                pose = resp.pose_stamped[0].pose
                
                x, y, z = pose.position.x, pose.position.y, pose.position.z
                
                q = pose.orientation
                rot = R.from_quat([q.x, q.y, q.z, q.w])
                roll, pitch, yaw = rot.as_euler('xyz', degrees=False)
                
                # ✅ UPDATED: NOW SAVING ALL 15 ELEMENTS (Joints, XYZ, RPY, Quaternions)
                results.append([
                    samples[i][0], samples[i][1], samples[i][2], samples[i][3], samples[i][4],
                    x, y, z, 
                    roll, pitch, yaw, 
                    q.x, q.y, q.z, q.w
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ✅ UPDATED: SAVING 15 COLUMNS TO CSV
        columns =[
            'j1', 'j2', 'j3', 'j4', 'j5', 
            'x', 'y', 'z', 
            'roll', 'pitch', 'yaw', 
            'qx', 'qy', 'qz', 'qw'
        ]
        poses_df = pd.DataFrame(data[:, :15], columns=columns)
        
        # Save to a fixed filename so the Sampler script can easily find it
        poses_file = "omx_all_reachable_poses_6dof_FULL.csv"
        poses_df.to_csv(poses_file, index=False)

        print(f"💾 Saved {valid_poses_count} FULL poses (Joints+XYZ+RPY+Quat) to: {poses_file}")

def main(args=None):
    rclpy.init(args=args)
    analyzer = AdvancedWorkspaceAnalyzer()
    analyzer.run_advanced_analysis(num_samples=8000)
    analyzer.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()