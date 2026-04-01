#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetMotionPlan, GetPositionFK
from moveit_msgs.msg import MotionPlanRequest, Constraints
from moveit_msgs.msg import OrientationConstraint, PositionConstraint
from moveit_msgs.msg import MoveItErrorCodes
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
# import os

import yaml
import numpy as np
import pandas as pd
import os
import time
import math
from scipy.spatial.transform import Rotation as R


class RollBenchmarkNode(Node):

    def __init__(self):
        super().__init__('roll_benchmark_node')

        # Parameters
        self.declare_parameter('arm_type', '5dof')
        self.declare_parameter('task_id', 0)
        self.declare_parameter('trials', 50)
        self.declare_parameter('roll_tol', 0.05)
        self.declare_parameter('pos_tol', 0.002)
        self.declare_parameter('tasks_file', 'tasks.yaml')

        self.arm_type = self.get_parameter('arm_type').value
        self.task_id = self.get_parameter('task_id').value
        self.n_trials = self.get_parameter('trials').value
        self.roll_tol = self.get_parameter('roll_tol').value
        self.pos_tol = self.get_parameter('pos_tol').value
        # tasks_file = self.get_parameter('tasks_file').value

        # # Load tasks
        # with open(tasks_file, 'r') as f:
        #     self.tasks = yaml.safe_load(f)['tasks']
        # Resolve tasks.yaml from package share directory
        pkg_share = get_package_share_directory('metric_pkg')
        tasks_file_param = self.get_parameter('tasks_file').value

        tasks_file = os.path.join(pkg_share, 'config', tasks_file_param)

        self.get_logger().info(f"Loading tasks from: {tasks_file}")

        with open(tasks_file, 'r') as f:
            self.tasks = yaml.safe_load(f)['tasks']

        self.current_task = self.tasks[self.task_id]

        # Services
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')

        self.get_logger().info("Waiting for MoveIt services...")
        self.plan_client.wait_for_service()
        self.fk_client.wait_for_service()
        self.get_logger().info("Services connected.")

        # Results file
        os.makedirs("results", exist_ok=True)
        self.results_file = f"results/benchmark_{self.arm_type}_task{self.task_id}.csv"
        self.results = []

    # ----------------------------
    # Constraint Creation
    # ----------------------------
    def create_constraints(self, target_pose):

        constraints = Constraints()

        # Position constraint
        pc = PositionConstraint()
        pc.header.frame_id = "link1"
        pc.link_name = "end_effector_link"

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.pos_tol]

        pc.constraint_region.primitives.append(sphere)
        pc.constraint_region.primitive_poses.append(target_pose.pose)
        pc.weight = 1.0
        constraints.position_constraints.append(pc)

        # Orientation constraint (tight roll)
        oc = OrientationConstraint()
        oc.header.frame_id = "link1"
        oc.link_name = "end_effector_link"
        oc.orientation = target_pose.pose.orientation

        oc.absolute_x_axis_tolerance = 1.0
        oc.absolute_y_axis_tolerance = 1.0
        oc.absolute_z_axis_tolerance = self.roll_tol
        oc.weight = 1.0

        constraints.orientation_constraints.append(oc)

        return constraints

    # ----------------------------
    # Orientation Error (Proper)
    # ----------------------------
    def compute_orientation_error(self, actual_q, target_q):

        r_actual = R.from_quat(actual_q)
        r_target = R.from_quat(target_q)

        relative = r_target.inv() * r_actual
        angle = relative.magnitude()

        return abs(angle)

    # ----------------------------
    # Failure Taxonomy
    # ----------------------------
    def classify_failure(self, error_code):

        if error_code == MoveItErrorCodes.SUCCESS:
            return "success"

        elif error_code == MoveItErrorCodes.NO_IK_SOLUTION:
            return "ik_failure"

        elif error_code == MoveItErrorCodes.PLANNING_FAILED:
            return "planning_failed"

        elif error_code == MoveItErrorCodes.TIMED_OUT:
            return "timeout"

        else:
            return "other"

    # ----------------------------
    # Get Final FK
    # ----------------------------
    def get_fk(self):

        req = GetPositionFK.Request()
        req.header.frame_id = "link1"
        req.fk_link_names = ["end_effector_link"]
        req.robot_state.joint_state = JointState()

        future = self.fk_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()

        if result is None:
            return None

        return result.pose_stamped[0]

    # ----------------------------
    # Run Benchmark
    # ----------------------------
    def run(self):

        self.get_logger().info(f"Running {self.n_trials} trials...")

        for trial in range(self.n_trials):

            target_pose = PoseStamped()
            target_pose.header.frame_id = "link1"

            target_pose.pose.position.x = self.current_task["goal_pose"]["position"]["x"]
            target_pose.pose.position.y = self.current_task["goal_pose"]["position"]["y"]
            target_pose.pose.position.z = self.current_task["goal_pose"]["position"]["z"]

            target_pose.pose.orientation.x = self.current_task["goal_pose"]["orientation"]["x"]
            target_pose.pose.orientation.y = self.current_task["goal_pose"]["orientation"]["y"]
            target_pose.pose.orientation.z = self.current_task["goal_pose"]["orientation"]["z"]
            target_pose.pose.orientation.w = self.current_task["goal_pose"]["orientation"]["w"]

            request = GetMotionPlan.Request()
            mpr = MotionPlanRequest()

            mpr.group_name = "manipulator"
            mpr.allowed_planning_time = 5.0
            mpr.num_planning_attempts = 5
            mpr.goal_constraints.append(self.create_constraints(target_pose))

            request.motion_plan_request = mpr

            future = self.plan_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)

            response = future.result()

            if response is None:
                self.results.append([trial, 0, "service_failure", 0, np.nan, np.nan])
                continue

            error_code = response.motion_plan_response.error_code.val
            planning_time = response.motion_plan_response.planning_time
            # failure_type = self.classify_failure(error_code)
            self.get_logger().info(f"Raw MoveIt error code: {error_code}")
            failure_type = self.classify_failure(error_code)

            if failure_type != "success":
                self.results.append([trial, 0, failure_type, planning_time, np.nan, np.nan])
                continue

            # Check final pose
            final_pose = self.get_fk()

            if final_pose is None:
                self.results.append([trial, 0, "fk_failure", planning_time, np.nan, np.nan])
                continue

            pos_err = math.sqrt(
                (final_pose.pose.position.x - target_pose.pose.position.x) ** 2 +
                (final_pose.pose.position.y - target_pose.pose.position.y) ** 2 +
                (final_pose.pose.position.z - target_pose.pose.position.z) ** 2
            )

            actual_q = [
                final_pose.pose.orientation.x,
                final_pose.pose.orientation.y,
                final_pose.pose.orientation.z,
                final_pose.pose.orientation.w
            ]

            target_q = [
                target_pose.pose.orientation.x,
                target_pose.pose.orientation.y,
                target_pose.pose.orientation.z,
                target_pose.pose.orientation.w
            ]

            roll_err = self.compute_orientation_error(actual_q, target_q)

            success = pos_err < self.pos_tol and roll_err < self.roll_tol

            self.results.append([
                trial,
                int(success),
                "success" if success else "pose_error",
                planning_time,
                pos_err,
                roll_err
            ])

        df = pd.DataFrame(self.results,
                          columns=["trial", "success", "failure_type",
                                   "planning_time", "position_error",
                                   "orientation_error"])

        df.to_csv(self.results_file, index=False)
        self.get_logger().info(f"Saved results to {self.results_file}")


def main():
    rclpy.init()
    node = RollBenchmarkNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()