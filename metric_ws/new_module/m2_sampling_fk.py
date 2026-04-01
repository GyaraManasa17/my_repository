import rclpy
import random
import csv
import time
import datetime
import json
import math

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

        # 1. Setup FK Client
        self.fk_client = node.create_client(GetPositionFK, "/compute_fk")
        self.node.get_logger().info("⏳ Waiting for MoveIt FK service...")
        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("❌ FK service '/compute_fk' not available!")
        self.node.get_logger().info("✅ FK service connected.")

        # 2. Setup State Validity Client (For Collision Checking)
        self.validity_client = node.create_client(GetStateValidity, "/check_state_validity")
        self.node.get_logger().info("⏳ Waiting for MoveIt State Validity service...")
        if not self.validity_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("❌ Validity service '/check_state_validity' not available!")
        self.node.get_logger().info("✅ Validity service connected.")

        # Setup standard robot state message (For single synchronous calls)
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

    # =========================================================================
    # PUBLIC SYNCHRONOUS METHODS (Kept intact so connected modules don't break)
    # =========================================================================

    def check_state_validity(self, joint_values):
        """Check if a single state is self-colliding or violates constraints."""
        request = GetStateValidity.Request()
        self.joint_state.position = joint_values
        self.robot_state.joint_state = self.joint_state
        request.robot_state = self.robot_state
        request.group_name = self.group_name

        future = self.validity_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

        if not future.done():
            self.node.get_logger().warning("⚠️ State validity check timed out.")
            return False

        return future.result().valid

    def compute_fk(self, joint_values):
        """Calculate the 3D pose of the end-effector given specific joint angles."""
        request = GetPositionFK.Request()
        self.joint_state.position = joint_values
        self.robot_state.joint_state = self.joint_state
        request.robot_state = self.robot_state
        request.fk_link_names =[self.end_link]

        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

        if not future.done():
            return None

        response = future.result()
        if response is None or len(response.pose_stamped) == 0:
            return None

        return response.pose_stamped[0].pose

    def is_pose_valid(self, pose):
        """Validates the calculated metrics (Checks for NaNs and normalizes Quaternions)."""
        if pose is None:
            return False
            
        pos =[pose.position.x, pose.position.y, pose.position.z]
        ori =[pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        
        if any(math.isnan(v) or math.isinf(v) for v in pos + ori):
            self.node.get_logger().warning("⚠️ FK returned NaN or Inf values!")
            return False
            
        norm = math.sqrt(sum(q**2 for q in ori))
        if abs(norm - 1.0) > 1e-3:
            self.node.get_logger().warning(f"⚠️ FK returned unnormalized quaternion (norm: {norm:.4f})!")
            return False
            
        return True

    # =========================================================================
    # INTERNAL ASYNCHRONOUS BATCH METHODS (For blazing fast dataset generation)
    # =========================================================================

    def _create_robot_state(self, joint_values):
        """Helper to create a fresh RobotState for batch requests."""
        rs = RobotState()
        rs.joint_state.name = self.joints
        rs.joint_state.position = joint_values
        return rs

    def _check_state_validity_batch(self, joint_values_list):
        """Asynchronously evaluate multiple joint states for collision."""
        futures =[]
        for jv in joint_values_list:
            req = GetStateValidity.Request()
            req.robot_state = self._create_robot_state(jv)
            req.group_name = self.group_name
            futures.append((jv, self.validity_client.call_async(req)))

        # Spin node until all async requests in this batch complete
        while rclpy.ok() and any(not f[1].done() for f in futures):
            rclpy.spin_once(self.node, timeout_sec=0.05)

        valid_joints =[]
        for jv, f in futures:
            res = f[1].result()
            if res is not None and res.valid:
                valid_joints.append(jv)
        return valid_joints

    def _compute_fk_batch(self, joint_values_list):
        """Asynchronously evaluate FK for multiple joint states."""
        futures =[]
        for jv in joint_values_list:
            req = GetPositionFK.Request()
            req.robot_state = self._create_robot_state(jv)
            req.fk_link_names = [self.end_link]
            futures.append((jv, self.fk_client.call_async(req)))

        # Spin node until all async requests in this batch complete
        while rclpy.ok() and any(not f[1].done() for f in futures):
            rclpy.spin_once(self.node, timeout_sec=0.05)

        results =[]
        for jv, f in futures:
            res = f[1].result()
            if res is not None and len(res.pose_stamped) > 0:
                results.append((jv, res.pose_stamped[0].pose))
            else:
                results.append((jv, None))
        return results

    # =========================================================================
    # MAIN DATASET GENERATOR
    # =========================================================================

    def sample_workspace(self, n_samples=10000, seed=None, output_prefix="workspace_fk_dataset"):
        if seed is not None:
            random.seed(seed)
            self.node.get_logger().info(f"🎲 Random seed set to {seed}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_file = f"{output_prefix}_{timestamp}.csv"
        metadata_file = f"{output_prefix}_{timestamp}_metadata.json"

        self.node.get_logger().info(f"📁 Dataset file: {dataset_file}")
        self.node.get_logger().info(f"⚡ FAST MODE: Generating exactly {n_samples} valid, collision-free samples...")

        # 1. Save Metadata
        metadata = {
            "robot_dof": self.dof,
            "base_link": self.base_link,
            "end_link": self.end_link,
            "joint_names": self.joints,
            "joint_limits": self.limits,
            "num_samples": n_samples,
            "timestamp": timestamp,
            "seed": seed,
            "collision_checked": True,
            "batch_processed": True
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

        # 2. Setup CSV Columns
        fieldnames =["x", "y", "z", "qx", "qy", "qz", "qw"]
        for j in self.joints:
            fieldnames.append(j)

        # Metrics Trackers
        valid_samples_collected = 0
        total_attempts = 0
        collision_free_count = 0
        fk_success_count = 0
        
        start_time = time.time()
        next_progress_milestone = max(1, n_samples // 10)
        BATCH_SIZE = 100 # Capped batch size to prevent overloading MoveIt

        # 3. Generate Data using Async Batches
        with open(dataset_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while valid_samples_collected < n_samples:
                # Determine how many attempts we need this round
                needed = n_samples - valid_samples_collected
                batch_size = min(BATCH_SIZE, needed * 2) 
                batch_size = max(10, batch_size) # At least request 10 at a time

                batch_joints =[self.random_joint_configuration() for _ in range(batch_size)]
                total_attempts += batch_size

                # A. Asynchronous Collision Check
                valid_joints = self._check_state_validity_batch(batch_joints)
                collision_free_count += len(valid_joints)

                if not valid_joints:
                    continue  # Entire batch was in collision

                # B. Asynchronous FK Calculation
                fk_results = self._compute_fk_batch(valid_joints)

                # C. Validate Metrics and Save
                for jv, pose in fk_results:
                    if valid_samples_collected >= n_samples:
                        break # Stop immediately if we reached our goal

                    if pose is not None:
                        fk_success_count += 1
                        
                        if self.is_pose_valid(pose):
                            row = {}
                            for j, val in zip(self.joints, jv):
                                row[j] = val

                            row["x"], row["y"], row["z"] = pose.position.x, pose.position.y, pose.position.z
                            row["qx"], row["qy"] = pose.orientation.x, pose.orientation.y
                            row["qz"], row["qw"] = pose.orientation.z, pose.orientation.w

                            writer.writerow(row)
                            valid_samples_collected += 1

                            # Print progress gracefully
                            if valid_samples_collected >= next_progress_milestone or valid_samples_collected == n_samples:
                                progress = (valid_samples_collected / n_samples) * 100
                                self.node.get_logger().info(
                                    f"📊 Progress: {valid_samples_collected}/{n_samples} ({progress:.1f}%) "
                                    f"| Attempts so far: {total_attempts}"
                                )
                                next_progress_milestone += max(1, n_samples // 10)

        # 4. Final Analytics
        duration = time.time() - start_time
        efficiency = (valid_samples_collected / total_attempts) * 100 if total_attempts > 0 else 0.0
        collision_free_rate = (collision_free_count / total_attempts) * 100 if total_attempts > 0 else 0.0
        fk_rate = (fk_success_count / collision_free_count * 100) if collision_free_count > 0 else 0.0

        self.node.get_logger().info("\n✅ Workspace sampling complete!")
        self.node.get_logger().info(f"⏱ Duration: {duration:.2f} seconds")
        self.node.get_logger().info(f"📈 Total configurations attempted: {total_attempts}")
        self.node.get_logger().info(f"🎯 Valid Samples Saved: {valid_samples_collected}")
        
        # Missing metrics restored!
        self.node.get_logger().info(f"🛡️ Collision-Free Rate: {collision_free_rate:.2f}%")
        self.node.get_logger().info(f"✔ FK Success Rate (Out of collision-free): {fk_rate:.2f}%")
        self.node.get_logger().info(f"🏆 Overall Free-Space Efficiency: {efficiency:.2f}%")