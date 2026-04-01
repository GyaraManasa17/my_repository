import rclpy
import random
import csv
import time
import datetime
import json

from moveit_msgs.srv import GetPositionFK, GetPositionIK
from moveit_msgs.msg import RobotState, BoundingVolume
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose
from shape_msgs.msg import SolidPrimitive

from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint

class WorkspaceSampler:

    def __init__(self, node, parser):
        self.node = node
        self.parser = parser

        self.base_link = parser.base_link
        self.end_link = parser.end_link
        self.joints = parser.joints
        self.limits = parser.limits
        self.dof = parser.dof
        self.group_name = parser.group_name

        self.fk_client = node.create_client(GetPositionFK, "/compute_fk")
        self.ik_client = node.create_client(GetPositionIK, "/compute_ik")

        self.node.get_logger().info("⏳ Waiting for MoveIt FK service...")
        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("❌ FK service '/compute_fk' not available!")
        self.node.get_logger().info("✅ FK service connected.")

        self.node.get_logger().info("⏳ Waiting for MoveIt IK service...")
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("❌ IK service '/compute_ik' not available!")
        self.node.get_logger().info("✅ IK service connected.")

        self.robot_state = RobotState()
        self.joint_state = JointState()
        self.joint_state.name = self.joints
        self.robot_state.joint_state = self.joint_state

        self.move_group_client = ActionClient(node, MoveGroup, "/move_action")
        self.node.get_logger().info("⏳ Waiting for MoveIt planning action...")
        if not self.move_group_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("❌ MoveGroup planning action not available!")
        self.node.get_logger().info("✅ Motion planning action connected.")

    def random_joint_configuration(self):
        config =[]
        for low, high in self.limits:
            config.append(random.uniform(low, high))
        return config

    def compute_fk(self, joint_values):
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

    def test_motion_plan(self, pose):
        goal_msg = MoveGroup.Goal()
        request = MotionPlanRequest()
        request.group_name = self.group_name
        request.allowed_planning_time = 2.0  # ✅ Give the planner some specific time

        constraints = Constraints()

        # ✅ FIX 1: Provide a valid geometric volume (a tiny 1cm sphere) for the target
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self.base_link
        pos_constraint.link_name = self.end_link
        
        target_volume = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]  # 1cm radius tolerance
        
        sphere_pose = Pose()
        sphere_pose.position.x = pose.position.x
        sphere_pose.position.y = pose.position.y
        sphere_pose.position.z = pose.position.z
        sphere_pose.orientation.w = 1.0
        
        target_volume.primitives.append(sphere)
        target_volume.primitive_poses.append(sphere_pose)
        
        pos_constraint.constraint_region = target_volume
        pos_constraint.weight = 1.0

        # ✅ FIX 2: Set reasonable tolerances so the planner doesn't require infinite precision
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = self.base_link
        ori_constraint.link_name = self.end_link
        ori_constraint.orientation = pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.1  # ~5.7 degrees tolerance
        ori_constraint.absolute_y_axis_tolerance = 0.1
        ori_constraint.absolute_z_axis_tolerance = 0.1
        ori_constraint.weight = 1.0

        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint)

        request.goal_constraints.append(constraints)
        goal_msg.request = request

        future = self.move_group_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.0)

        if not future.done():
            return False, "PLANNING_TIMEOUT"

        goal_handle = future.result()
        if not goal_handle.accepted:
            return False, "PLANNING_REJECTED"

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=5.0)

        if not result_future.done():
            return False, "PLANNING_TIMEOUT"

        result = result_future.result()
        if result.result.error_code.val != 1:
            return False, "PLANNING_FAILED"

        return True, "SUCCESS"

    def compute_ik(self, pose):
        request = GetPositionIK.Request()
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_link
        pose_stamped.pose = pose

        request.ik_request.group_name = self.group_name
        request.ik_request.pose_stamped = pose_stamped
        request.ik_request.timeout.sec = 1

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

        if not future.done():
            return False, "IK_TIMEOUT"

        response = future.result()
        if response.error_code.val != 1:
            return False, "IK_NO_SOLUTION"

        return True, "SUCCESS"

    def sample_workspace(self, n_samples=10000, seed=None, output_prefix="workspace_dataset"):
        if seed is not None:
            random.seed(seed)
            self.node.get_logger().info(f"🎲 Random seed set to {seed}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_file = f"{output_prefix}_{timestamp}.csv"
        metadata_file = f"{output_prefix}_{timestamp}_metadata.json"

        self.node.get_logger().info(f"📁 Dataset file: {dataset_file}")

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

        fieldnames =["x", "y", "z", "qx", "qy", "qz", "qw"]
        for j in self.joints:
            fieldnames.append(j)
        fieldnames +=["fk_success", "ik_success", "planning_success", "failure_reason"]

        success_fk = 0
        success_ik = 0
        success_plan = 0
        failure_stats = {}

        start_time = time.time()

        with open(dataset_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(n_samples):
                joint_values = self.random_joint_configuration()
                pose = self.compute_fk(joint_values)

                row = {}
                if pose is not None:
                    row["x"], row["y"], row["z"] = pose.position.x, pose.position.y, pose.position.z
                    row["qx"], row["qy"] = pose.orientation.x, pose.orientation.y
                    row["qz"], row["qw"] = pose.orientation.z, pose.orientation.w
                    row["fk_success"] = 1
                    success_fk += 1

                    ik_success, reason = self.compute_ik(pose)
                    row["ik_success"] = int(ik_success)

                    if ik_success:
                        success_ik += 1
                        plan_success, plan_reason = self.test_motion_plan(pose)
                        row["planning_success"] = int(plan_success)

                        if plan_success:
                            success_plan += 1
                            row["failure_reason"] = "SUCCESS"
                        else:
                            row["failure_reason"] = plan_reason
                            failure_stats[plan_reason] = failure_stats.get(plan_reason, 0) + 1
                    else:
                        row["planning_success"] = 0
                        row["failure_reason"] = reason
                        failure_stats[reason] = failure_stats.get(reason, 0) + 1
                else:
                    row["x"] = row["y"] = row["z"] = None
                    row["qx"] = row["qy"] = row["qz"] = row["qw"] = None
                    row["fk_success"] = row["ik_success"] = row["planning_success"] = 0
                    row["failure_reason"] = "FK_FAILED"
                    failure_stats["FK_FAILED"] = failure_stats.get("FK_FAILED", 0) + 1

                for j, val in zip(self.joints, joint_values):
                    row[j] = val

                writer.writerow(row)

                if (i + 1) % 10 == 0 or i == n_samples - 1:
                    progress = (i + 1) / n_samples * 100
                    self.node.get_logger().info(f"📊 Sample {i+1}/{n_samples} ({progress:.1f}%)")

        duration = time.time() - start_time
        self.node.get_logger().info("✅ Workspace sampling complete.")
        self.node.get_logger().info(f"⏱ Duration: {duration:.2f} seconds")
        self.node.get_logger().info(f"✔ FK Success Rate: {(success_fk / n_samples) * 100:.2f}%")
        self.node.get_logger().info(f"✔ IK Success Rate: {(success_ik / n_samples) * 100:.2f}%")
        self.node.get_logger().info(f"✔ Planning Success Rate: {(success_plan / n_samples) * 100:.2f}%")
        
        self.node.get_logger().info("📉 Failure Breakdown:")
        for reason, count in failure_stats.items():
            percent = (count / n_samples) * 100
            self.node.get_logger().info(f"   {reason}: {count} ({percent:.2f}%)")






# # autocert_modules/workspace_sampler.py

# import rclpy
# import random
# import csv
# import time
# import datetime
# import json

# from moveit_msgs.srv import GetPositionFK, GetPositionIK
# from moveit_msgs.msg import RobotState
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import PoseStamped

# from moveit_msgs.action import MoveGroup
# from rclpy.action import ActionClient
# from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint


# class WorkspaceSampler:

#     def __init__(self, node, parser):

#         self.node = node
#         self.parser = parser

#         self.base_link = parser.base_link
#         self.end_link = parser.end_link
#         self.joints = parser.joints
#         self.limits = parser.limits
#         self.dof = parser.dof
#         self.group_name = parser.group_name

#         # FK client
#         self.fk_client = node.create_client(GetPositionFK, "/compute_fk")

#         # IK client
#         self.ik_client = node.create_client(GetPositionIK, "/compute_ik")


#         self.node.get_logger().info("⏳ Waiting for MoveIt FK service...")

#         if not self.fk_client.wait_for_service(timeout_sec=10.0):
#             raise RuntimeError("❌ FK service '/compute_fk' not available!")

#         self.node.get_logger().info("✅ FK service connected.")

#         self.node.get_logger().info("⏳ Waiting for MoveIt IK service...")

#         if not self.ik_client.wait_for_service(timeout_sec=10.0):
#             raise RuntimeError("❌ IK service '/compute_ik' not available!")

#         self.node.get_logger().info("✅ IK service connected.")

#         # Reusable state objects
#         self.robot_state = RobotState()
#         self.joint_state = JointState()

#         self.joint_state.name = self.joints
#         self.robot_state.joint_state = self.joint_state

#         # Motion Planning Action Client
#         self.move_group_client = ActionClient(
#             node,
#             MoveGroup,
#             "/move_action"
#         )

#         self.node.get_logger().info("⏳ Waiting for MoveIt planning action...")

#         if not self.move_group_client.wait_for_server(timeout_sec=10.0):
#             raise RuntimeError("❌ MoveGroup planning action not available!")

#         self.node.get_logger().info("✅ Motion planning action connected.")
#     # ----------------------------------------------------------

#     def random_joint_configuration(self):

#         config = []

#         for low, high in self.limits:
#             config.append(random.uniform(low, high))

#         return config

#     # ----------------------------------------------------------

#     def compute_fk(self, joint_values):

#         request = GetPositionFK.Request()

#         self.joint_state.position = joint_values

#         request.robot_state = self.robot_state
#         request.fk_link_names = [self.end_link]

#         future = self.fk_client.call_async(request)

#         rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

#         if not future.done():
#             return None

#         response = future.result()

#         if response is None:
#             return None

#         if len(response.pose_stamped) == 0:
#             return None

#         return response.pose_stamped[0].pose

#     # ----------------------------------------------------------

#     def test_motion_plan(self, pose):

#         goal_msg = MoveGroup.Goal()

#         request = MotionPlanRequest()

#         request.group_name = self.group_name

#         constraints = Constraints()

#         pos_constraint = PositionConstraint()
#         pos_constraint.link_name = self.end_link
#         pos_constraint.target_point_offset.x = pose.position.x
#         pos_constraint.target_point_offset.y = pose.position.y
#         pos_constraint.target_point_offset.z = pose.position.z

#         ori_constraint = OrientationConstraint()
#         ori_constraint.link_name = self.end_link
#         ori_constraint.orientation = pose.orientation

#         constraints.position_constraints.append(pos_constraint)
#         constraints.orientation_constraints.append(ori_constraint)

#         request.goal_constraints.append(constraints)

#         goal_msg.request = request

#         future = self.move_group_client.send_goal_async(goal_msg)

#         rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.0)

#         if not future.done():
#             return False, "PLANNING_TIMEOUT"

#         goal_handle = future.result()

#         if not goal_handle.accepted:
#             return False, "PLANNING_REJECTED"

#         result_future = goal_handle.get_result_async()

#         rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=5.0)

#         if not result_future.done():
#             return False, "PLANNING_TIMEOUT"

#         result = result_future.result()

#         if result.result.error_code.val != 1:
#             return False, "PLANNING_FAILED"

#         return True, "SUCCESS"

#     def compute_ik(self, pose):

#         request = GetPositionIK.Request()

#         pose_stamped = PoseStamped()
#         pose_stamped.header.frame_id = self.base_link
#         pose_stamped.pose = pose

#         request.ik_request.group_name = self.group_name
#         request.ik_request.pose_stamped = pose_stamped
#         request.ik_request.timeout.sec = 1

#         future = self.ik_client.call_async(request)

#         rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

#         if not future.done():
#             return False, "IK_TIMEOUT"

#         response = future.result()

#         if response.error_code.val != 1:
#             return False, "IK_NO_SOLUTION"

#         return True, "SUCCESS"

#     # ----------------------------------------------------------

#     def sample_workspace(self, n_samples=10000, seed=None, output_prefix="workspace_dataset"):

#         if seed is not None:
#             random.seed(seed)
#             self.node.get_logger().info(f"🎲 Random seed set to {seed}")

#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#         dataset_file = f"{output_prefix}_{timestamp}.csv"
#         metadata_file = f"{output_prefix}_{timestamp}_metadata.json"

#         self.node.get_logger().info(f"📁 Dataset file: {dataset_file}")

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

#         self.node.get_logger().info(f"🧾 Metadata saved to {metadata_file}")

#         # CSV Headers

#         fieldnames = [
#             "x", "y", "z",
#             "qx", "qy", "qz", "qw"
#         ]

#         for j in self.joints:
#             fieldnames.append(j)

#         fieldnames += [
#             "fk_success",
#             "ik_success",
#             "planning_success",
#             "failure_reason"
#         ]

#         success_fk = 0
#         success_ik = 0
#         success_plan = 0
#         failure_stats = {}

#         start_time = time.time()

#         with open(dataset_file, "w", newline="") as f:

#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()

#             for i in range(n_samples):

#                 joint_values = self.random_joint_configuration()

#                 pose = self.compute_fk(joint_values)

#                 row = {}

#                 if pose is not None:

#                     row["x"] = pose.position.x
#                     row["y"] = pose.position.y
#                     row["z"] = pose.position.z

#                     row["qx"] = pose.orientation.x
#                     row["qy"] = pose.orientation.y
#                     row["qz"] = pose.orientation.z
#                     row["qw"] = pose.orientation.w

#                     row["fk_success"] = 1
#                     success_fk += 1

#                     ik_success, reason = self.compute_ik(pose)

#                     row["ik_success"] = int(ik_success)

#                     # if ik_success:
#                     #     success_ik += 1
#                     #     row["planning_success"] = 1
#                     #     row["failure_reason"] = "SUCCESS"
#                     if ik_success:

#                         success_ik += 1

#                         plan_success, plan_reason = self.test_motion_plan(pose)

#                         row["planning_success"] = int(plan_success)

#                         if plan_success:
#                             success_plan+=1
#                             row["failure_reason"] = "SUCCESS"
#                         else:
#                             row["failure_reason"] = plan_reason
#                             failure_stats[plan_reason] = failure_stats.get(plan_reason, 0) + 1
#                     else:
#                         row["planning_success"] = 0
#                         row["failure_reason"] = reason
#                         failure_stats[reason] = failure_stats.get(reason, 0) + 1

#                 else:

#                     row["x"] = None
#                     row["y"] = None
#                     row["z"] = None

#                     row["qx"] = None
#                     row["qy"] = None
#                     row["qz"] = None
#                     row["qw"] = None

#                     row["fk_success"] = 0
#                     row["ik_success"] = 0
#                     row["planning_success"] = 0
#                     row["failure_reason"] = "FK_FAILED"
#                     failure_stats["FK_FAILED"] = failure_stats.get("FK_FAILED", 0) + 1

#                 for j, val in zip(self.joints, joint_values):
#                     row[j] = val

#                 writer.writerow(row)

#                 if (i + 1) % 500 == 0 or i == n_samples - 1:

#                     progress = (i + 1) / n_samples * 100

#                     self.node.get_logger().info(
#                         f"📊 Sample {i+1}/{n_samples} ({progress:.1f}%)"
#                     )

#         duration = time.time() - start_time

#         self.node.get_logger().info("✅ Workspace sampling complete.")

#         self.node.get_logger().info(f"⏱ Duration: {duration:.2f} seconds")

#         self.node.get_logger().info(
#             f"✔ FK Success Rate: {(success_fk / n_samples) * 100:.2f}%"
#         )

#         self.node.get_logger().info(
#             f"✔ IK Success Rate: {(success_ik / n_samples) * 100:.2f}%"
#         )

#         self.node.get_logger().info(
#             f"✔ Planning Success Rate: {(success_plan / n_samples) * 100:.2f}%"
#         )

#         self.node.get_logger().info("📉 Failure Breakdown:")

#         for reason, count in failure_stats.items():
#             percent = (count / n_samples) * 100
#             self.node.get_logger().info(
#                 f"   {reason}: {count} ({percent:.2f}%)"
#             )




# # # autocert_modules/workspace_sampler.py

# # import rclpy
# # import random
# # import csv
# # import time
# # import datetime
# # import json

# # from moveit_msgs.srv import GetPositionFK
# # from moveit_msgs.msg import RobotState
# # from sensor_msgs.msg import JointState


# # class WorkspaceSampler:

# #     def __init__(self, node, parser):

# #         self.node = node
# #         self.parser = parser

# #         self.base_link = parser.base_link
# #         self.end_link = parser.end_link
# #         self.joints = parser.joints
# #         self.limits = parser.limits
# #         self.dof = parser.dof

# #         # FK client
# #         self.fk_client = node.create_client(GetPositionFK, "/compute_fk")

# #         self.node.get_logger().info("⏳ Waiting for MoveIt FK service...")

# #         if not self.fk_client.wait_for_service(timeout_sec=10.0):
# #             raise RuntimeError("❌ FK service '/compute_fk' not available!")

# #         self.node.get_logger().info("✅ FK service connected.")

# #         # -------------------------------------------------
# #         # Reusable message objects (performance improvement)
# #         # -------------------------------------------------

# #         self.robot_state = RobotState()
# #         self.joint_state = JointState()

# #         self.joint_state.name = self.joints
# #         self.robot_state.joint_state = self.joint_state

# #     # ----------------------------------------------------------

# #     def random_joint_configuration(self):

# #         config = []

# #         for low, high in self.limits:
# #             config.append(random.uniform(low, high))

# #         return config

# #     # ----------------------------------------------------------

# #     def compute_fk(self, joint_values):

# #         request = GetPositionFK.Request()

# #         # Reuse joint state object
# #         self.joint_state.position = joint_values

# #         request.robot_state = self.robot_state
# #         request.fk_link_names = [self.end_link]

# #         future = self.fk_client.call_async(request)

# #         rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

# #         if not future.done():
# #             self.node.get_logger().warning("⚠️ FK service timeout.")
# #             return None

# #         response = future.result()

# #         if response is None:
# #             return None

# #         if len(response.pose_stamped) == 0:
# #             return None

# #         return response.pose_stamped[0].pose

# #     # ----------------------------------------------------------

# #     def sample_workspace(self, n_samples=10000, seed=None, output_prefix="workspace_dataset"):

# #         if seed is not None:
# #             random.seed(seed)
# #             self.node.get_logger().info(f"🎲 Random seed set to {seed}")

# #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# #         dataset_file = f"{output_prefix}_{timestamp}.csv"
# #         metadata_file = f"{output_prefix}_{timestamp}_metadata.json"

# #         self.node.get_logger().info(f"📁 Dataset file: {dataset_file}")

# #         # -----------------------
# #         # Metadata
# #         # -----------------------

# #         metadata = {

# #             "robot_dof": self.dof,
# #             "base_link": self.base_link,
# #             "end_link": self.end_link,
# #             "joint_names": self.joints,
# #             "joint_limits": self.limits,
# #             "num_samples": n_samples,
# #             "timestamp": timestamp,
# #             "seed": seed

# #         }

# #         with open(metadata_file, "w") as f:
# #             json.dump(metadata, f, indent=4)

# #         self.node.get_logger().info(f"🧾 Metadata saved to {metadata_file}")

# #         # -----------------------
# #         # CSV Headers
# #         # -----------------------

# #         fieldnames = [
# #             "x", "y", "z",
# #             "qx", "qy", "qz", "qw"
# #         ]

# #         for j in self.joints:
# #             fieldnames.append(j)

# #         fieldnames.append("fk_success")

# #         success_count = 0

# #         start_time = time.time()

# #         # -----------------------
# #         # CSV Streaming
# #         # -----------------------

# #         with open(dataset_file, "w", newline="") as f:

# #             writer = csv.DictWriter(f, fieldnames=fieldnames)
# #             writer.writeheader()

# #             for i in range(n_samples):

# #                 joint_values = self.random_joint_configuration()

# #                 pose = self.compute_fk(joint_values)

# #                 row = {}

# #                 if pose is not None:

# #                     row["x"] = pose.position.x
# #                     row["y"] = pose.position.y
# #                     row["z"] = pose.position.z

# #                     row["qx"] = pose.orientation.x
# #                     row["qy"] = pose.orientation.y
# #                     row["qz"] = pose.orientation.z
# #                     row["qw"] = pose.orientation.w

# #                     row["fk_success"] = 1

# #                     success_count += 1

# #                 else:

# #                     row["x"] = None
# #                     row["y"] = None
# #                     row["z"] = None

# #                     row["qx"] = None
# #                     row["qy"] = None
# #                     row["qz"] = None
# #                     row["qw"] = None

# #                     row["fk_success"] = 0

# #                 for j, val in zip(self.joints, joint_values):
# #                     row[j] = val

# #                 writer.writerow(row)

# #                 # -----------------------
# #                 # Reduced logging frequency
# #                 # -----------------------

# #                 if (i + 1) % 500 == 0 or i == n_samples - 1:

# #                     progress = (i + 1) / n_samples * 100

# #                     self.node.get_logger().info(
# #                         f"📊 Sample {i+1}/{n_samples} ({progress:.1f}%)"
# #                     )

# #         duration = time.time() - start_time

# #         self.node.get_logger().info("✅ Workspace sampling complete.")

# #         self.node.get_logger().info(f"⏱ Duration: {duration:.2f} seconds")

# #         self.node.get_logger().info(
# #             f"✔ FK Success Rate: {(success_count / n_samples) * 100:.2f}%"
# #         )







# # # autocert_modules/workspace_sampler.py

# # import rclpy
# # import random
# # import csv
# # import math
# # import time
# # import datetime
# # import json

# # from moveit_msgs.srv import GetPositionFK
# # from moveit_msgs.msg import RobotState
# # from sensor_msgs.msg import JointState
# # from geometry_msgs.msg import PoseStamped


# # class WorkspaceSampler:

# #     def __init__(self, node, parser):

# #         self.node = node
# #         self.parser = parser

# #         self.base_link = parser.base_link
# #         self.end_link = parser.end_link
# #         self.joints = parser.joints
# #         self.limits = parser.limits
# #         self.dof = parser.dof

# #         # FK service client
# #         self.fk_client = node.create_client(GetPositionFK, "/compute_fk")

# #         self.node.get_logger().info("⏳ Waiting for MoveIt FK service...")

# #         if not self.fk_client.wait_for_service(timeout_sec=10.0):
# #             raise RuntimeError("❌ FK service '/compute_fk' not available!")

# #         self.node.get_logger().info("✅ FK service connected.")

# #     # ----------------------------------------------------------

# #     def random_joint_configuration(self):

# #         config = []

# #         for low, high in self.limits:
# #             value = random.uniform(low, high)
# #             config.append(value)

# #         return config

# #     # ----------------------------------------------------------

# #     def compute_fk(self, joint_values):

# #         request = GetPositionFK.Request()

# #         robot_state = RobotState()
# #         joint_state = JointState()

# #         joint_state.name = self.joints
# #         joint_state.position = joint_values

# #         robot_state.joint_state = joint_state

# #         request.robot_state = robot_state
# #         request.fk_link_names = [self.end_link]

# #         future = self.fk_client.call_async(request)

# #         rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

# #         if not future.done():
# #             self.node.get_logger().warning("⚠️ FK service timeout.")
# #             return None

# #         response = future.result()

# #         if response is None:
# #             return None

# #         if len(response.pose_stamped) == 0:
# #             return None

# #         return response.pose_stamped[0].pose

# #     # ----------------------------------------------------------

# #     def sample_workspace(self, n_samples=2000, seed=None, output_prefix="workspace_dataset"):

# #         if seed is not None:
# #             random.seed(seed)
# #             self.node.get_logger().info(f"🎲 Random seed set to {seed}")

# #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# #         dataset_file = f"{output_prefix}_{timestamp}.csv"
# #         metadata_file = f"{output_prefix}_{timestamp}_metadata.json"

# #         self.node.get_logger().info(f"📁 Dataset file: {dataset_file}")

# #         # -----------------------
# #         # Metadata file
# #         # -----------------------

# #         metadata = {

# #             "robot_dof": self.dof,
# #             "base_link": self.base_link,
# #             "end_link": self.end_link,
# #             "joint_names": self.joints,
# #             "joint_limits": self.limits,
# #             "num_samples": n_samples,
# #             "timestamp": timestamp,
# #             "seed": seed

# #         }

# #         with open(metadata_file, "w") as f:
# #             json.dump(metadata, f, indent=4)

# #         self.node.get_logger().info(f"🧾 Metadata saved to {metadata_file}")

# #         # -----------------------
# #         # CSV headers
# #         # -----------------------

# #         fieldnames = [

# #             "x", "y", "z",
# #             "qx", "qy", "qz", "qw"

# #         ]

# #         for j in self.joints:
# #             fieldnames.append(j)

# #         fieldnames.append("fk_success")

# #         # -----------------------
# #         # CSV streaming write
# #         # -----------------------

# #         success_count = 0

# #         start_time = time.time()

# #         with open(dataset_file, "w", newline="") as f:

# #             writer = csv.DictWriter(f, fieldnames=fieldnames)

# #             writer.writeheader()

# #             for i in range(n_samples):

# #                 joint_values = self.random_joint_configuration()

# #                 pose = self.compute_fk(joint_values)

# #                 row = {}

# #                 if pose is not None:

# #                     row["x"] = pose.position.x
# #                     row["y"] = pose.position.y
# #                     row["z"] = pose.position.z

# #                     row["qx"] = pose.orientation.x
# #                     row["qy"] = pose.orientation.y
# #                     row["qz"] = pose.orientation.z
# #                     row["qw"] = pose.orientation.w

# #                     row["fk_success"] = 1

# #                     success_count += 1

# #                 else:

# #                     row["x"] = None
# #                     row["y"] = None
# #                     row["z"] = None

# #                     row["qx"] = None
# #                     row["qy"] = None
# #                     row["qz"] = None
# #                     row["qw"] = None

# #                     row["fk_success"] = 0

# #                 for j, val in zip(self.joints, joint_values):
# #                     row[j] = val

# #                 writer.writerow(row)

# #                 # Progress reporting
# #                 progress = (i + 1) / n_samples * 100

# #                 if (i + 1) % 50 == 0 or i == n_samples - 1:
# #                     self.node.get_logger().info(
# #                         f"📊 Sample {i+1}/{n_samples} ({progress:.1f}%)"
# #                     )

# #         duration = time.time() - start_time

# #         self.node.get_logger().info("✅ Workspace sampling complete.")

# #         self.node.get_logger().info(f"⏱ Duration: {duration:.2f} seconds")

# #         self.node.get_logger().info(
# #             f"✔ FK Success Rate: {(success_count / n_samples) * 100:.2f}%"
# #         )





# # import rclpy
# # from moveit_msgs.srv import GetPositionFK
# # from moveit_msgs.msg import RobotState
# # from sensor_msgs.msg import JointState
# # import random
# # import csv


# # class WorkspaceSampler:

# #     def __init__(self, node, parser):

# #         self.node = node
# #         self.parser = parser

# #         self.base_link = parser.base_link
# #         self.end_link = parser.end_link
# #         self.joints = parser.joints
# #         self.limits = parser.limits
# #         self.dof = parser.dof

# #         # FK Service Client
# #         self.fk_client = node.create_client(GetPositionFK, "/compute_fk")

# #         self.node.get_logger().info("⏳ Waiting for MoveIt FK service...")
# #         self.fk_client.wait_for_service()
# #         self.node.get_logger().info("✅ FK service connected.")

# #     # ------------------------------------------------

# #     def random_configuration(self):
# #         """Generate random joint configuration within limits"""

# #         config = []

# #         for lower, upper in self.limits:
# #             config.append(random.uniform(lower, upper))

# #         return config

# #     # ------------------------------------------------

# #     def compute_fk(self, joint_positions):
# #         """Call MoveIt FK service"""

# #         request = GetPositionFK.Request()

# #         request.header.frame_id = self.base_link
# #         request.fk_link_names = [self.end_link]

# #         robot_state = RobotState()
# #         joint_state = JointState()

# #         joint_state.name = self.joints
# #         joint_state.position = joint_positions

# #         robot_state.joint_state = joint_state
# #         request.robot_state = robot_state

# #         future = self.fk_client.call_async(request)
# #         rclpy.spin_until_future_complete(self.node, future)

# #         response = future.result()

# #         if response.error_code.val != 1:
# #             return None

# #         pose = response.pose_stamped[0].pose

# #         return {
# #             "x": pose.position.x,
# #             "y": pose.position.y,
# #             "z": pose.position.z,
# #             "qx": pose.orientation.x,
# #             "qy": pose.orientation.y,
# #             "qz": pose.orientation.z,
# #             "qw": pose.orientation.w
# #         }

# #     # ------------------------------------------------

# #     def sample_workspace(self, n_samples=2000, output_file="workspace_dataset.csv"):

# #         self.node.get_logger().info(f"🚀 Generating {n_samples} workspace samples")

# #         dataset = []
# #         success_count = 0

# #         for i in range(n_samples):

# #             config = self.random_configuration()

# #             fk_result = self.compute_fk(config)

# #             row = {}

# #             # store joint configuration
# #             for i_j, name in enumerate(self.joints):
# #                 row[name] = config[i_j]

# #             if fk_result is not None:

# #                 row.update(fk_result)
# #                 row["fk_success"] = 1
# #                 success_count += 1

# #             else:

# #                 row["x"] = None
# #                 row["y"] = None
# #                 row["z"] = None
# #                 row["qx"] = None
# #                 row["qy"] = None
# #                 row["qz"] = None
# #                 row["qw"] = None
# #                 row["fk_success"] = 0

# #             dataset.append(row)

# #             if i % 100 == 0:
# #                 self.node.get_logger().info(f"Sample {i}/{n_samples}")

# #         success_rate = success_count / n_samples

# #         self.node.get_logger().info(
# #             f"✅ FK Success Rate: {success_rate*100:.2f}%"
# #         )

# #         # -------------------------
# #         # Save CSV
# #         # -------------------------

# #         fieldnames = (
# #             ["x", "y", "z", "qx", "qy", "qz", "qw"]
# #             + self.joints
# #             + ["fk_success"]
# #         )

# #         with open(output_file, "w", newline="") as f:

# #             writer = csv.DictWriter(f, fieldnames=fieldnames)
# #             writer.writeheader()

# #             for row in dataset:
# #                 writer.writerow(row)

# #         self.node.get_logger().info(
# #             f"📁 Dataset saved → {output_file}"
# #         )