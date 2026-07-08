#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK
from geometry_msgs.msg import Pose
import numpy as np
import pandas as pd
import math
import argparse
import sys

class DatasetGenerator(Node):
    def __init__(self):
        super().__init__("dataset_generator")
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        
        # Wait for service with better error handling
        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ FK service /compute_fk not available")
            self.get_logger().error("   Check if MoveIt is running!")
            return
        self.get_logger().info("✅ FK service ready")

    def generate_dataset(self, num_trials=1000, seed=42):
        """Generate dataset of random joint configurations and their corresponding end-effector poses"""
        
        np.random.seed(seed)
        
        # 6DOF joint names from URDF (always use 6DOF model)
        joint_names = [
            "joint1",           # -π to π
            "joint2",           # -1.5 to 1.5  
            "joint3",           # -1.5 to 1.4
            "joint4",           # -1.7 to 1.97
            "joint5_roll"       # -π to π
        ]
        
        # Joint limits from URDF
        joint_limits = [
            (-math.pi, math.pi),    # joint1
            (-1.5, 1.5),           # joint2
            (-1.5, 1.4),           # joint3
            (-1.7, 1.97),          # joint4
            (-math.pi, math.pi)    # joint5_roll
        ]
        
        data = []
        end_link = "wrist_roll_link"  # 6DOF end effector
        
        self.get_logger().info(f"🚀 Generating {num_trials} pose samples...")
        
        successful_trials = 0
        for trial_idx in range(num_trials):
            # Generate random joint values within URDF limits
            joint_values = []
            for i, (lower, upper) in enumerate(joint_limits):
                joint_values.append(np.random.uniform(lower, upper))
            
            # Call /compute_fk service
            fk_req = GetPositionFK.Request()
            fk_req.header.frame_id = "link1"
            fk_req.fk_link_names = [end_link]
            fk_req.robot_state.joint_state.name = joint_names
            fk_req.robot_state.joint_state.position = joint_values
            
            future = self.fk_client.call_async(fk_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            
            if future.result() is None:
                self.get_logger().warn(f"Trial {trial_idx}: FK service failed")
                continue
                
            fk_resp = future.result()
            if len(fk_resp.pose_stamped) == 0:
                self.get_logger().warn(f"Trial {trial_idx}: No pose returned")
                continue
            
            # Extract pose (x, y, z, qx, qy, qz, qw)
            pose = fk_resp.pose_stamped[0].pose
            pose_data = [
                pose.position.x,
                pose.position.y, 
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            ]
            
            data.append([trial_idx] + joint_values + pose_data)
            successful_trials += 1
            
            if successful_trials % 50 == 0:
                self.get_logger().info(f"✅ {successful_trials}/{num_trials} successful trials")
        
        if len(data) == 0:
            self.get_logger().error("❌ No successful trials! Check MoveIt setup.")
            return pd.DataFrame()
        
        # Create DataFrame with all columns
        columns = ["trial"] + joint_names + ["x", "y", "z", "qx", "qy", "qz", "qw"]
        df = pd.DataFrame(data, columns=columns)
        
        # Save complete dataset
        complete_filename = f"generated_goals_6dof_{len(df)}_samples.csv"
        df.to_csv(complete_filename, index=False)
        
        # Save poses only (exactly as you specified for loading)
        pose_df = df[["x", "y", "z", "qx", "qy", "qz", "qw"]]
        pose_filename = "generated_goals.csv"  # Exact filename you requested
        pose_df.to_csv(pose_filename, index=False)
        
        self.get_logger().info(f"🎉 Dataset COMPLETE!")
        self.get_logger().info(f"📊 Full joints+poses: {complete_filename} ({len(df)} samples)")
        self.get_logger().info(f"🎯 LOAD THIS for planning: {pose_filename}")
        
        return df

def main():
    rclpy.init()
    
    parser = argparse.ArgumentParser(description="Generate 6DOF pose dataset")
    parser.add_argument("--trials", type=int, default=1000, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator()
    
    # Only generate if service was found
    if hasattr(generator, 'fk_client') and generator.fk_client:
        df = generator.generate_dataset(num_trials=args.trials, seed=args.seed)
        if not df.empty:
            print(f"\n🎊 SUCCESS: Generated {len(df)} valid 6DOF samples!")
            print(f"💾 Files ready for PHASE 1!")
        else:
            print("\n❌ No valid samples. Check MoveIt!")
    else:
        print("\n❌ MoveIt FK service /compute_fk not available!")
        print("   Run: ros2 launch open_manipulator_x_moveit_config demo.launch.py")
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()
