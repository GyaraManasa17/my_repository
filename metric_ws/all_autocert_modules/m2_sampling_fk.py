import rclpy
import random
import csv
import time
import datetime
import json

from moveit_msgs.srv import GetPositionFK
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

        # Only create the FK client
        self.fk_client = node.create_client(GetPositionFK, "/compute_fk")

        self.node.get_logger().info("⏳ Waiting for MoveIt FK service...")
        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("❌ FK service '/compute_fk' not available!")
        self.node.get_logger().info("✅ FK service connected.")

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

    def compute_fk(self, joint_values):
        """Calculate the 3D pose of the end-effector given specific joint angles."""
        request = GetPositionFK.Request()
        self.joint_state.position = joint_values
        request.robot_state = self.robot_state
        request.fk_link_names = [self.end_link]

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
        dataset_file = f"{output_prefix}_{timestamp}.csv"
        metadata_file = f"{output_prefix}_{timestamp}_metadata.json"

        self.node.get_logger().info(f"📁 Dataset file: {dataset_file}")

        # 1. Save Metadata
        metadata = {
            "robot_dof": self.dof,
            "base_link": self.base_link,
            "end_link": self.end_link,
            "joint_names": self.joints,
            "joint_limits": self.limits,
            "num_samples": n_samples,
            "timestamp": timestamp,
            "seed": seed
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

        # 2. Setup CSV Columns (Only Pose and Joints)
        fieldnames =["x", "y", "z", "qx", "qy", "qz", "qw"]
        for j in self.joints:
            fieldnames.append(j)

        success_fk = 0
        start_time = time.time()

        # 3. Generate Data and Save to CSV
        with open(dataset_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(n_samples):
                joint_values = self.random_joint_configuration()
                pose = self.compute_fk(joint_values)

                row = {}
                
                # Add joint values to the row
                for j, val in zip(self.joints, joint_values):
                    row[j] = val

                # Add Cartesian Pose to the row
                if pose is not None:
                    row["x"], row["y"], row["z"] = pose.position.x, pose.position.y, pose.position.z
                    row["qx"], row["qy"] = pose.orientation.x, pose.orientation.y
                    row["qz"], row["qw"] = pose.orientation.z, pose.orientation.w
                    success_fk += 1
                else:
                    # In the rare case FK fails, write None
                    row["x"] = row["y"] = row["z"] = None
                    row["qx"] = row["qy"] = row["qz"] = row["qw"] = None

                writer.writerow(row)

                # Print progress every 10% or at the end
                if (i + 1) % max(1, (n_samples // 10)) == 0 or i == n_samples - 1:
                    progress = (i + 1) / n_samples * 100
                    self.node.get_logger().info(f"📊 Sample {i+1}/{n_samples} ({progress:.1f}%)")

        # 4. Final Analytics
        duration = time.time() - start_time
        self.node.get_logger().info("✅ FK Workspace sampling complete.")
        self.node.get_logger().info(f"⏱ Duration: {duration:.2f} seconds")
        self.node.get_logger().info(f"✔ FK Success Rate: {(success_fk / n_samples) * 100:.2f}%")