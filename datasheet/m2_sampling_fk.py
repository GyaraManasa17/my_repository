import rclpy
import random
import csv
import time
import datetime
import json
import os  # NEW: Added for directory management

from moveit_msgs.srv import GetPositionFK, GetStateValidity
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose

class WorkspaceSampler:

    def __init__(self, node, parser):
        self.node = node
        self.parser = parser

        # Extracting robot details
        self.base_link = parser.base_link
        self.end_link = parser.end_link
        self.joints = parser.joints
        self.limits = parser.limits
        self.dof = parser.dof
        self.group_name = parser.group_name

        # ---------------------------------------------------------
        # 1. Connect to FK Service
        # ---------------------------------------------------------
        self.fk_client = node.create_client(GetPositionFK, "/compute_fk")
        self.node.get_logger().info("⏳ Waiting for MoveIt FK service...")
        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("❌ FK service '/compute_fk' not available!")
        self.node.get_logger().info("✅ FK service connected.")

        # ---------------------------------------------------------
        # 2. Connect to State Validity (Collision) Service
        # ---------------------------------------------------------
        self.validity_client = node.create_client(GetStateValidity, "/check_state_validity")
        self.node.get_logger().info("⏳ Waiting for MoveIt State Validity (Collision) service...")
        if not self.validity_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("❌ Validity service '/check_state_validity' not available!")
        self.node.get_logger().info("✅ Collision Checking service connected.")

        # ---------------------------------------------------------
        # 3. NEW: Setup Metrics Directory
        # ---------------------------------------------------------
        self.output_dir = "fk_metrics"
        os.makedirs(self.output_dir, exist_ok=True)
        self.node.get_logger().info(f"📁 Output directory verified: ./{self.output_dir}/")

        # Setup standard robot state message
        self.robot_state = RobotState()
        self.joint_state = JointState()
        self.joint_state.name = self.joints
        self.robot_state.joint_state = self.joint_state

    def random_joint_configuration(self):
        """Generate random joint values within the physical limits."""
        config =[]
        for low, high in self.limits:
            config.append(random.uniform(low, high))
        return config

    def is_state_valid(self, joint_values):
        """Check if the joint configuration is collision-free."""
        req = GetStateValidity.Request()
        self.joint_state.position = joint_values
        req.robot_state = self.robot_state
        req.group_name = self.group_name

        future = self.validity_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

        if not future.done():
            self.node.get_logger().warning("⚠️ Collision check timed out. Assuming invalid.")
            return False

        res = future.result()
        return res.valid

    def compute_fk(self, joint_values):
        """Calculate the 3D pose of the end-effector given specific joint angles."""
        request = GetPositionFK.Request()
        self.joint_state.position = joint_values
        request.robot_state = self.robot_state
        request.fk_link_names =[self.end_link]
        
        request.header.frame_id = self.base_link 

        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

        if not future.done():
            return None

        response = future.result()
        if response is None or len(response.pose_stamped) == 0:
            return None

        return response.pose_stamped[0].pose

    def sample_workspace(self, n_samples=10000, seed=None, output_prefix="workspace_fk_dataset"):
        if seed is not None:
            random.seed(seed)
            self.node.get_logger().info(f"🎲 Random seed set to {seed}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # NEW: Route all output files into the fk_metrics folder
        dataset_file = os.path.join(self.output_dir, f"{output_prefix}_{timestamp}.csv")
        metadata_file = os.path.join(self.output_dir, f"{output_prefix}_{timestamp}_metadata.json")
        metrics_file = os.path.join(self.output_dir, f"fk_performance_metrics_{timestamp}.json")

        self.node.get_logger().info(f"📁 Saving dataset to: {dataset_file}")

        # Setup CSV Columns
        fieldnames =["x", "y", "z", "qx", "qy", "qz", "qw"]
        for j in self.joints:
            fieldnames.append(j)

        start_time = time.time()
        
        # ---------------------------------------------------------
        # NEW: Advanced tracking metrics
        # ---------------------------------------------------------
        valid_samples = 0
        rejected_samples = 0
        total_attempts = 0
        fk_attempts = 0
        fk_successes = 0
        max_attempts = n_samples * 50  # Prevent infinite loop

        with open(dataset_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while valid_samples < n_samples and total_attempts < max_attempts:
                total_attempts += 1
                joint_values = self.random_joint_configuration()

                # --- STEP 1: REJECTION SAMPLING (COLLISION CHECK) ---
                if not self.is_state_valid(joint_values):
                    rejected_samples += 1
                    continue  

                # --- STEP 2: COMPUTE FK FOR VALID STATES ---
                fk_attempts += 1
                pose = self.compute_fk(joint_values)
                if pose is None:
                    continue 
                fk_successes += 1

                # --- STEP 3: SAVE TO DATASET ---
                row = {}
                for j, val in zip(self.joints, joint_values):
                    row[j] = val

                row["x"], row["y"], row["z"] = pose.position.x, pose.position.y, pose.position.z
                row["qx"], row["qy"] = pose.orientation.x, pose.orientation.y
                row["qz"], row["qw"] = pose.orientation.z, pose.orientation.w

                writer.writerow(row)
                valid_samples += 1

                # Print progress every 10%
                if valid_samples % max(1, (n_samples // 10)) == 0:
                    progress = (valid_samples / n_samples) * 100
                    self.node.get_logger().info(f"📊 Valid Samples {valid_samples}/{n_samples} ({progress:.0f}%) | Rejected: {rejected_samples}")

        # ---------------------------------------------------------
        # NEW: Calculate Scientific Metrics
        # ---------------------------------------------------------
        duration = time.time() - start_time
        rejection_rate = (rejected_samples / total_attempts) * 100 if total_attempts > 0 else 0
        
        # Reachability Ratio: The percentage of the mathematical joint space that is physically reachable without collision.
        reachability_ratio = (valid_samples / total_attempts) * 100 if total_attempts > 0 else 0
        
        # FK Success Rate: The percentage of times the Forward Kinematics solver successfully returned a pose for a valid joint state.
        fk_success_rate = (fk_successes / fk_attempts) * 100 if fk_attempts > 0 else 0

        # Save Metadata
        metadata = {
            "robot_dof": self.dof,
            "base_link": self.base_link,
            "end_link": self.end_link,
            "joint_names": self.joints,
            "joint_limits": self.limits,
            "timestamp": timestamp,
            "seed": seed
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

        # Save FK Metrics
        metrics = {
            "sampling_duration_seconds": round(duration, 3),
            "target_samples": n_samples,
            "total_configurations_tested": total_attempts,
            "valid_collision_free_poses": valid_samples,
            "rejected_colliding_poses": rejected_samples,
            "reachability_ratio_percent": round(reachability_ratio, 2),
            "collision_rejection_rate_percent": round(rejection_rate, 2),
            "fk_computation_attempts": fk_attempts,
            "fk_computation_successes": fk_successes,
            "fk_success_rate_percent": round(fk_success_rate, 2)
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        # Final Analytics Display
        self.node.get_logger().info("✅ ============================================")
        self.node.get_logger().info("✅ Workspace Sampling Complete!")
        self.node.get_logger().info(f"⏱ Duration: {duration:.2f} seconds")
        self.node.get_logger().info(f"✔ Valid Poses Saved: {valid_samples}")
        self.node.get_logger().info(f"❌ Collisions Rejected: {rejected_samples}")
        self.node.get_logger().info(f"📈 Reachability Ratio: {reachability_ratio:.2f}% (Collision-Free Space)")
        self.node.get_logger().info(f"📉 Rejection Rate: {rejection_rate:.2f}% (Colliding Space)")
        self.node.get_logger().info(f"🧮 FK Solver Success Rate: {fk_success_rate:.2f}%")
        self.node.get_logger().info(f"📁 All files saved successfully in ./{self.output_dir}/")
        self.node.get_logger().info("✅ ============================================")