import rclpy
import random
import csv
import time
import datetime
import json

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

        # Create the FK client
        self.fk_client = node.create_client(GetPositionFK, "/compute_fk")

        self.node.get_logger().info("⏳ Waiting for MoveIt FK service...")
        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("❌ FK service '/compute_fk' not available!")
        self.node.get_logger().info("✅ FK service connected.")

        # Create the State Validity client for collision checking
        self.validity_client = node.create_client(GetStateValidity, "/check_state_validity")

        self.node.get_logger().info("⏳ Waiting for MoveIt State Validity service...")
        if not self.validity_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("❌ State Validity service '/check_state_validity' not available!")
        self.node.get_logger().info("✅ State Validity service connected.")

        # Setup standard robot state message
        self.robot_state = RobotState()
        self.joint_state = JointState()
        self.joint_state.name = self.joints
        self.robot_state.joint_state = self.joint_state

    def random_joint_configuration(self):
        """Generate random joint values within the physical limits."""
        config = []
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

    def check_self_collision(self, joint_values):
        """Check if the given joint configuration results in a self-collision."""
        request = GetStateValidity.Request()
        self.joint_state.position = joint_values
        request.robot_state = self.robot_state
        request.group_name = self.group_name

        future = self.validity_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

        if not future.done():
            return None

        response = future.result()
        if response is None:
            return None

        # If valid is False (and limits are respected), it means it's colliding
        return not response.valid

    # ---> MODIFIED: Added z_table_limit <---
    def sample_workspace(self, n_samples=10000, seed=None, output_prefix="workspace_fk_dataset", z_table_limit=0.0):
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
            "seed": seed,
            "z_table_limit": z_table_limit,          # <--- ADDED
            "collisions_ignored": False              # <--- ADDED
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

        # 2. Setup CSV Columns
        fieldnames = ["x", "y", "z", "qx", "qy", "qz", "qw", "is_colliding"]
        for j in self.joints:
            fieldnames.append(j)

        start_time = time.time()
        
        # ---> MODIFIED Tracking Variables <---
        valid_samples = 0
        total_attempts = 0
        below_table_count = 0
        collision_count = 0

        # 3. Generate Data and Save to CSV
        with open(dataset_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # ---> MODIFIED: Changed 'for' to 'while' loop <---
            while valid_samples < n_samples:
                total_attempts += 1
                joint_values = self.random_joint_configuration()
                
                # Compute FK First
                pose = self.compute_fk(joint_values)
                
                if pose is None:
                    continue

                # ---> ADDED: Discard if below table <---
                if pose.position.z < z_table_limit:
                    below_table_count += 1
                    continue

                # ---> ADDED: Discard if colliding <---
                is_colliding = self.check_self_collision(joint_values)
                if is_colliding:
                    collision_count += 1
                    continue

                # If we reached here, the pose is VALID, NON-COLLIDING, and ABOVE TABLE!
                row = {}
                
                # Add joint values to the row
                for j, val in zip(self.joints, joint_values):
                    row[j] = val

                # Hardcode to False because we are filtering out all True ones
                row["is_colliding"] = False

                # Add Cartesian Pose to the row
                row["x"], row["y"], row["z"] = pose.position.x, pose.position.y, pose.position.z
                row["qx"], row["qy"] = pose.orientation.x, pose.orientation.y
                row["qz"], row["qw"] = pose.orientation.z, pose.orientation.w

                writer.writerow(row)
                valid_samples += 1

                # Print progress every 10%
                if valid_samples % max(1, (n_samples // 10)) == 0:
                    progress = (valid_samples / n_samples) * 100
                    self.node.get_logger().info(f"📊 Secured {valid_samples}/{n_samples} safe poses ({progress:.1f}%) [Attempts: {total_attempts}]")

        # 4. Final Analytics
        duration = time.time() - start_time
        self.node.get_logger().info("✅ FK Workspace sampling complete.")
        self.node.get_logger().info(f"⏱ Duration: {duration:.2f} seconds")
        self.node.get_logger().info(f"💥 Rejected (Self-Collision): {collision_count}")
        self.node.get_logger().info(f"⬇️ Rejected (Below Table Z<{z_table_limit}): {below_table_count}")



# import rclpy
# import random
# import csv
# import time
# import datetime
# import json

# from moveit_msgs.srv import GetPositionFK, GetStateValidity
# from moveit_msgs.msg import RobotState
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import PoseStamped, Pose

# class WorkspaceSampler:

#     def __init__(self, node, parser):
#         self.node = node
#         self.parser = parser

#         # Extracting robot details
#         self.base_link = parser.base_link
#         self.end_link = parser.end_link
#         self.joints = parser.joints
#         self.limits = parser.limits
#         self.dof = parser.dof
#         self.group_name = parser.group_name

#         # Create the FK client
#         self.fk_client = node.create_client(GetPositionFK, "/compute_fk")

#         self.node.get_logger().info("⏳ Waiting for MoveIt FK service...")
#         if not self.fk_client.wait_for_service(timeout_sec=10.0):
#             raise RuntimeError("❌ FK service '/compute_fk' not available!")
#         self.node.get_logger().info("✅ FK service connected.")

#         # Create the State Validity client for collision checking
#         self.validity_client = node.create_client(GetStateValidity, "/check_state_validity")

#         self.node.get_logger().info("⏳ Waiting for MoveIt State Validity service...")
#         if not self.validity_client.wait_for_service(timeout_sec=10.0):
#             raise RuntimeError("❌ State Validity service '/check_state_validity' not available!")
#         self.node.get_logger().info("✅ State Validity service connected.")

#         # Setup standard robot state message
#         self.robot_state = RobotState()
#         self.joint_state = JointState()
#         self.joint_state.name = self.joints
#         self.robot_state.joint_state = self.joint_state

#     def random_joint_configuration(self):
#         """Generate random joint values within the physical limits."""
#         config = []
#         for low, high in self.limits:
#             config.append(random.uniform(low, high))
#         return config

#     def compute_fk(self, joint_values):
#         """Calculate the 3D pose of the end-effector given specific joint angles."""
#         request = GetPositionFK.Request()
#         self.joint_state.position = joint_values
#         request.robot_state = self.robot_state
#         request.fk_link_names = [self.end_link]

#         future = self.fk_client.call_async(request)
#         rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

#         if not future.done():
#             return None

#         response = future.result()
#         if response is None or len(response.pose_stamped) == 0:
#             return None

#         return response.pose_stamped[0].pose

#     def check_self_collision(self, joint_values):
#         """Check if the given joint configuration results in a self-collision."""
#         request = GetStateValidity.Request()
#         self.joint_state.position = joint_values
#         request.robot_state = self.robot_state
#         request.group_name = self.group_name

#         future = self.validity_client.call_async(request)
#         rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

#         if not future.done():
#             return None

#         response = future.result()
#         if response is None:
#             return None

#         # If valid is False (and limits are respected), it means it's colliding
#         return not response.valid

#     def sample_workspace(self, n_samples=10000, seed=None, output_prefix="workspace_fk_dataset"):
#         if seed is not None:
#             random.seed(seed)
#             self.node.get_logger().info(f"🎲 Random seed set to {seed}")

#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         dataset_file = f"{output_prefix}_{timestamp}.csv"
#         metadata_file = f"{output_prefix}_{timestamp}_metadata.json"

#         self.node.get_logger().info(f"📁 Dataset file: {dataset_file}")

#         # 1. Save Metadata
#         metadata = {
#             "robot_dof": self.dof,
#             "base_link": self.base_link,
#             "end_link": self.end_link,
#             "joint_names": self.joints,
#             "joint_limits": self.limits,
#             "num_samples": n_samples,
#             "timestamp": timestamp,
#             "seed": seed
#         }

#         with open(metadata_file, "w") as f:
#             json.dump(metadata, f, indent=4)

#         # 2. Setup CSV Columns (Added is_colliding)
#         fieldnames = ["x", "y", "z", "qx", "qy", "qz", "qw", "is_colliding"]
#         for j in self.joints:
#             fieldnames.append(j)

#         success_fk = 0
#         start_time = time.time()

#         # 3. Generate Data and Save to CSV
#         with open(dataset_file, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()

#             for i in range(n_samples):
#                 joint_values = self.random_joint_configuration()
#                 pose = self.compute_fk(joint_values)
#                 is_colliding = self.check_self_collision(joint_values)

#                 row = {}
                
#                 # Add joint values to the row
#                 for j, val in zip(self.joints, joint_values):
#                     row[j] = val

#                 # Add Collision State to the row
#                 row["is_colliding"] = is_colliding

#                 # Add Cartesian Pose to the row
#                 if pose is not None:
#                     row["x"], row["y"], row["z"] = pose.position.x, pose.position.y, pose.position.z
#                     row["qx"], row["qy"] = pose.orientation.x, pose.orientation.y
#                     row["qz"], row["qw"] = pose.orientation.z, pose.orientation.w
#                     success_fk += 1
#                 else:
#                     # In the rare case FK fails, write None
#                     row["x"] = row["y"] = row["z"] = None
#                     row["qx"] = row["qy"] = row["qz"] = row["qw"] = None

#                 writer.writerow(row)

#                 # Print progress every 10% or at the end
#                 if (i + 1) % max(1, (n_samples // 10)) == 0 or i == n_samples - 1:
#                     progress = (i + 1) / n_samples * 100
#                     self.node.get_logger().info(f"📊 Sample {i+1}/{n_samples} ({progress:.1f}%)")

#         # 4. Final Analytics
#         duration = time.time() - start_time
#         self.node.get_logger().info("✅ FK Workspace sampling complete.")
#         self.node.get_logger().info(f"⏱ Duration: {duration:.2f} seconds")
#         self.node.get_logger().info(f"✔ FK Success Rate: {(success_fk / n_samples) * 100:.2f}%")


# import rclpy
# import random
# import csv
# import time
# import datetime
# import json

# from moveit_msgs.srv import GetPositionFK
# from moveit_msgs.msg import RobotState
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import PoseStamped, Pose

# class WorkspaceSampler:

#     def __init__(self, node, parser):
#         self.node = node
#         self.parser = parser

#         # Extracting robot details
#         self.base_link = parser.base_link
#         self.end_link = parser.end_link
#         self.joints = parser.joints
#         self.limits = parser.limits
#         self.dof = parser.dof
#         self.group_name = parser.group_name

#         # Only create the FK client
#         self.fk_client = node.create_client(GetPositionFK, "/compute_fk")

#         self.node.get_logger().info("⏳ Waiting for MoveIt FK service...")
#         if not self.fk_client.wait_for_service(timeout_sec=10.0):
#             raise RuntimeError("❌ FK service '/compute_fk' not available!")
#         self.node.get_logger().info("✅ FK service connected.")

#         # Setup standard robot state message
#         self.robot_state = RobotState()
#         self.joint_state = JointState()
#         self.joint_state.name = self.joints
#         self.robot_state.joint_state = self.joint_state

#     def random_joint_configuration(self):
#         """Generate random joint values within the physical limits."""
#         config =[]
#         for low, high in self.limits:
#             config.append(random.uniform(low, high))
#         return config

#     def compute_fk(self, joint_values):
#         """Calculate the 3D pose of the end-effector given specific joint angles."""
#         request = GetPositionFK.Request()
#         self.joint_state.position = joint_values
#         request.robot_state = self.robot_state
#         request.fk_link_names = [self.end_link]

#         future = self.fk_client.call_async(request)
#         rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

#         if not future.done():
#             return None

#         response = future.result()
#         if response is None or len(response.pose_stamped) == 0:
#             return None

#         return response.pose_stamped[0].pose

#     def sample_workspace(self, n_samples=10000, seed=None, output_prefix="workspace_fk_dataset"):
#         if seed is not None:
#             random.seed(seed)
#             self.node.get_logger().info(f"🎲 Random seed set to {seed}")

#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         dataset_file = f"{output_prefix}_{timestamp}.csv"
#         metadata_file = f"{output_prefix}_{timestamp}_metadata.json"

#         self.node.get_logger().info(f"📁 Dataset file: {dataset_file}")

#         # 1. Save Metadata
#         metadata = {
#             "robot_dof": self.dof,
#             "base_link": self.base_link,
#             "end_link": self.end_link,
#             "joint_names": self.joints,
#             "joint_limits": self.limits,
#             "num_samples": n_samples,
#             "timestamp": timestamp,
#             "seed": seed
#         }

#         with open(metadata_file, "w") as f:
#             json.dump(metadata, f, indent=4)

#         # 2. Setup CSV Columns (Only Pose and Joints)
#         fieldnames =["x", "y", "z", "qx", "qy", "qz", "qw"]
#         for j in self.joints:
#             fieldnames.append(j)

#         success_fk = 0
#         start_time = time.time()

#         # 3. Generate Data and Save to CSV
#         with open(dataset_file, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()

#             for i in range(n_samples):
#                 joint_values = self.random_joint_configuration()
#                 pose = self.compute_fk(joint_values)

#                 row = {}
                
#                 # Add joint values to the row
#                 for j, val in zip(self.joints, joint_values):
#                     row[j] = val

#                 # Add Cartesian Pose to the row
#                 if pose is not None:
#                     row["x"], row["y"], row["z"] = pose.position.x, pose.position.y, pose.position.z
#                     row["qx"], row["qy"] = pose.orientation.x, pose.orientation.y
#                     row["qz"], row["qw"] = pose.orientation.z, pose.orientation.w
#                     success_fk += 1
#                 else:
#                     # In the rare case FK fails, write None
#                     row["x"] = row["y"] = row["z"] = None
#                     row["qx"] = row["qy"] = row["qz"] = row["qw"] = None

#                 writer.writerow(row)

#                 # Print progress every 10% or at the end
#                 if (i + 1) % max(1, (n_samples // 10)) == 0 or i == n_samples - 1:
#                     progress = (i + 1) / n_samples * 100
#                     self.node.get_logger().info(f"📊 Sample {i+1}/{n_samples} ({progress:.1f}%)")

#         # 4. Final Analytics
#         duration = time.time() - start_time
#         self.node.get_logger().info("✅ FK Workspace sampling complete.")
#         self.node.get_logger().info(f"⏱ Duration: {duration:.2f} seconds")
#         self.node.get_logger().info(f"✔ FK Success Rate: {(success_fk / n_samples) * 100:.2f}%")