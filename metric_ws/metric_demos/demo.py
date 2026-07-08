#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped


class SinglePoseTest(Node):
    def __init__(self):
        super().__init__("single_pose_test")

        self.client = self.create_client(GetMotionPlan, "/plan_kinematic_path")

        if not self.client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("MoveIt planning service not available")

    def send_goal(self):

        # -----------------------------
        # 🔹 HARD CODED TARGET POSITION
        # -----------------------------
        pose = PoseStamped()
        pose.header.frame_id = "link1"

        pose.pose.position.x = 0.25
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.20

        pose.pose.orientation.w = 1.0
        # -----------------------------

        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = "arm"
        req.motion_plan_request.allowed_planning_time = 5.0

        # 🔹 4 ACTIVE JOINTS ONLY
        req.motion_plan_request.start_state.joint_state.name = [
            "joint1", "joint2", "joint3", "joint4"
        ]
        req.motion_plan_request.start_state.joint_state.position = [0.0, 0.0, 0.0, 0.0]

        goal_constraints = Constraints()

        # Position constraint (3 mm sphere)
        pos_const = PositionConstraint()
        pos_const.header.frame_id = "link1"
        pos_const.link_name = "end_effector_link"

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.003]

        pos_const.constraint_region.primitives.append(sphere)
        pos_const.constraint_region.primitive_poses.append(pose.pose)
        pos_const.weight = 1.0

        goal_constraints.position_constraints.append(pos_const)

        # 🔹 Orientation relaxed heavily
        ori_const = OrientationConstraint()
        ori_const.header.frame_id = "link1"
        ori_const.link_name = "end_effector_link"
        ori_const.orientation = pose.pose.orientation
        ori_const.absolute_x_axis_tolerance = 3.14
        ori_const.absolute_y_axis_tolerance = 3.14
        ori_const.absolute_z_axis_tolerance = 3.14
        ori_const.weight = 0.1

        goal_constraints.orientation_constraints.append(ori_const)

        req.motion_plan_request.goal_constraints.append(goal_constraints)

        print("Sending planning request...")
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=6.0)

        if future.result() is None:
            print("❌ Service call failed")
            return

        resp = future.result().motion_plan_response

        if resp.error_code.val == resp.error_code.SUCCESS:
            print("✅ PLAN SUCCESSFUL")
        else:
            print("❌ PLAN FAILED")
            print("Error code:", resp.error_code.val)


def main():
    rclpy.init()
    node = SinglePoseTest()
    node.send_goal()
    rclpy.shutdown()


if __name__ == "__main__":
    main()