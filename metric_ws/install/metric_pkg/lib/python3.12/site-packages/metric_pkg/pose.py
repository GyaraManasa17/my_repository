#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetMotionPlan, GetPositionFK
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
import numpy as np
import pandas as pd
import time
import argparse


def generate_fixed_goals(num_trials, joint_names, seed=42):
    np.random.seed(seed)
    return [{j: np.random.uniform(-1.0, 1.0) for j in joint_names}
            for _ in range(num_trials)]


def generate_random_start(joint_names):
    return [np.random.uniform(-1.0, 1.0) for _ in joint_names]


class PlanningBenchmark(Node):
    def __init__(self):
        super().__init__("planning_benchmark_node")
        self.client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")

        if not self.client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("MoveIt planning service not available")

        if not self.fk_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("MoveIt FK service not available")


def run_trials(config_name,
               joint_names,
               trials,
               planning_time,
               pos_tol,
               ori_tol,
               random_start,
               seed):

    np.random.seed(seed)

    rclpy.init()
    node = PlanningBenchmark()
    trial_data = []

    fixed_goals = generate_fixed_goals(trials, joint_names, seed)

    print("\n========== POSE ROBUSTNESS BENCHMARK ==========")
    print(f"Config: {config_name}")
    print(f"Trials: {trials}")
    print(f"Planning time: {planning_time}")
    print(f"Position tol: {pos_tol}")
    print(f"Orientation tol: {ori_tol}")
    print(f"Random start: {random_start}")
    print("===============================================\n")

    if config_name=="6dof":
        end_link="wrist_roll_link"
    else:
        end_link="link5"

    for i in range(trials):

        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = "arm"
        req.motion_plan_request.num_planning_attempts = 3
        req.motion_plan_request.allowed_planning_time = planning_time

        # -------- START STATE --------
        req.motion_plan_request.start_state.joint_state.name = joint_names

        if random_start:
            req.motion_plan_request.start_state.joint_state.position = \
                generate_random_start(joint_names)
        else:
            req.motion_plan_request.start_state.joint_state.position = \
                [0.0] * len(joint_names)

        # -------- GOAL (POSE CONSTRAINT) --------
        goal_constraints = Constraints()
        current_goal = fixed_goals[i]

        # Use FK to generate reachable pose
        fk_req = GetPositionFK.Request()
        fk_req.header.frame_id = "link1"
        fk_req.fk_link_names = [end_link]
        fk_req.robot_state.joint_state.name = joint_names
        fk_req.robot_state.joint_state.position = \
            [current_goal[j] for j in joint_names]

        fk_future = node.fk_client.call_async(fk_req)
        rclpy.spin_until_future_complete(node, fk_future)
        fk_resp = fk_future.result()

        if fk_resp is None or len(fk_resp.pose_stamped) == 0:
            print(f"Trial {i}: FK failed.")
            continue

        goal_pose = fk_resp.pose_stamped[0]

        # Position constraint
        pos_const = PositionConstraint()
        pos_const.header.frame_id = "link1"
        pos_const.link_name = end_link

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [pos_tol]

        pos_const.constraint_region.primitives.append(sphere)
        pos_const.constraint_region.primitive_poses.append(goal_pose.pose)
        pos_const.weight = 1.0

        goal_constraints.position_constraints.append(pos_const)

        # Orientation constraint
        ori_const = OrientationConstraint()
        ori_const.header.frame_id = "link1"
        ori_const.link_name = end_link
        ori_const.orientation = goal_pose.pose.orientation
        ori_const.absolute_x_axis_tolerance = ori_tol
        ori_const.absolute_y_axis_tolerance = ori_tol
        ori_const.absolute_z_axis_tolerance = ori_tol
        ori_const.weight = 1.0

        goal_constraints.orientation_constraints.append(ori_const)

        req.motion_plan_request.goal_constraints.append(goal_constraints)

        # -------- EXECUTE --------
        start_time = time.time()
        future = node.client.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        wall_time = time.time() - start_time

        success = False
        planner_time = wall_time

        if future.result() is not None:
            resp = future.result().motion_plan_response
            success = (resp.error_code.val == 1)
            if success and resp.planning_time > 0:
                planner_time = resp.planning_time

        trial_data.append({
            "trial": i,
            "success": success,
            "planning_time": planner_time
        })

        if (i + 1) % 10 == 0:
            rate = np.mean([d["success"] for d in trial_data]) * 100
            print(f"{i+1}/{trials} - Success: {rate:.1f}%")

    rclpy.shutdown()

    df = pd.DataFrame(trial_data)
    filename = f"pose_benchmark_{config_name}_tol{ori_tol}_time{planning_time}.csv"
    df.to_csv(filename, index=False)

    print(f"\nSaved: {filename}")
    print(f"Final Success Rate: {df['success'].mean()*100:.2f}%\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True, choices=["5dof", "6dof"])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--planning_time", type=float, default=2.0)
    parser.add_argument("--pos_tol", type=float, default=0.003)
    parser.add_argument("--ori_tol", type=float, default=0.03)
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.config == "5dof":
        joint_names = ["joint1", "joint2", "joint3", "joint4"]
    else:
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5_roll"]

    run_trials(args.config,
               joint_names,
               args.trials,
               args.planning_time,
               args.pos_tol,
               args.ori_tol,
               args.random_start,
               args.seed)


if __name__ == "__main__":
    main()
