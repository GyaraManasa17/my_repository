#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, JointConstraint
import numpy as np
import pandas as pd
import argparse
import time
from datetime import datetime
import math

# ---------------------------
# Joint Limits (from URDF)
# ---------------------------
JOINT_LIMITS = {
    "joint1": (-math.pi, math.pi),
    "joint2": (-1.5, 1.5),
    "joint3": (-1.5, 1.4),
    "joint4": (-1.7, 1.97),
    "joint5_roll": (-math.pi, math.pi)
}

HOME_POSE = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 0.0,
    "joint5_roll": 0.0
}


# ---------------------------
# Generate Fixed Goals
# ---------------------------
def generate_fixed_goals(num_trials, seed=42):
    np.random.seed(seed)
    goals = []

    for _ in range(num_trials):
        goal = {}
        for joint, (low, high) in JOINT_LIMITS.items():
            goal[joint] = np.random.uniform(low, high)
        goals.append(goal)

    return goals


# ---------------------------
# Benchmark Node
# ---------------------------
class PlanningBenchmark(Node):

    def __init__(self):
        super().__init__("joint_space_benchmark")
        self.client = self.create_client(GetMotionPlan, "/plan_kinematic_path")

        if not self.client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("MoveIt planning service not available")


# ---------------------------
# Compute Joint-Space Path Length
# ---------------------------
def compute_path_length(trajectory, joint_names):
    points = trajectory.joint_trajectory.points
    if len(points) < 2:
        return 0.0

    total_length = 0.0

    for i in range(len(points) - 1):
        q1 = np.array(points[i].positions)
        q2 = np.array(points[i + 1].positions)
        total_length += np.linalg.norm(q2 - q1)

    return float(total_length)


# ---------------------------
# Main Trial Runner
# ---------------------------
def run_trials(config_name, trials):

    rclpy.init()
    node = PlanningBenchmark()

    if config_name == "5dof":
        joint_names = ["joint1", "joint2", "joint3", "joint4"]
    else:
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5_roll"]

    fixed_goals = generate_fixed_goals(trials)

    results = []

    print(f"\nRunning {trials} trials for {config_name} joint-space planning...\n")

    for i in range(trials):

        goal = fixed_goals[i]

        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = "arm"
        req.motion_plan_request.num_planning_attempts = 10
        req.motion_plan_request.allowed_planning_time = 5.0

        # Start from home pose
        req.motion_plan_request.start_state.joint_state.name = joint_names
        req.motion_plan_request.start_state.joint_state.position = [
            HOME_POSE[j] for j in joint_names
        ]

        goal_constraints = Constraints()

        for j in joint_names:
            jc = JointConstraint()
            jc.joint_name = j
            jc.position = goal[j]
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            goal_constraints.joint_constraints.append(jc)

        req.motion_plan_request.goal_constraints.append(goal_constraints)

        start_wall = time.time()
        future = node.client.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        wall_time = time.time() - start_wall

        success = False
        planning_time = None
        path_length = None
        traj_duration = None
        num_waypoints = None

        if future.result() is not None:
            resp = future.result().motion_plan_response
            success = (resp.error_code.val == 1)

            if success:
                planning_time = resp.planning_time if resp.planning_time > 0 else wall_time
                trajectory = resp.trajectory

                path_length = compute_path_length(trajectory, joint_names)

                points = trajectory.joint_trajectory.points
                num_waypoints = len(points)

                if points:
                    traj_duration = (
                        points[-1].time_from_start.sec +
                        points[-1].time_from_start.nanosec * 1e-9
                    )

        results.append({
            "trial": i,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "planning_time": planning_time,
            "path_length": path_length,
            "trajectory_duration": traj_duration,
            "num_waypoints": num_waypoints
        })

        print(f"Trial {i}: Success={success}")

    rclpy.shutdown()

    df = pd.DataFrame(results)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{config_name}_joint_{timestamp_str}.csv"
    df.to_csv(filename, index=False)

    print("\nSaved results to:", filename)
    print("Success Rate:", df["success"].mean() * 100, "%")
    print("Average Planning Time:", df["planning_time"].mean())


# ---------------------------
# Entry Point
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=["5dof", "6dof"])
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    run_trials(args.config, args.trials)


if __name__ == "__main__":
    main()
