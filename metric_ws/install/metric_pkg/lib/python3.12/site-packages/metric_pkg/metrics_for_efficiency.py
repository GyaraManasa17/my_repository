#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
import numpy as np
import pandas as pd
import time
import argparse
import json
from datetime import datetime, timezone
import math

class PoseBenchmark(Node):
    def __init__(self):
        super().__init__("pose_benchmark")
        self.client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        
        if not self.client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("MoveIt planning service not available")
        
        # Arm configurations
        self.configs = {
            "6dof": {
                "joints": ["joint1", "joint2", "joint3", "joint4", "joint5_roll"],
                "end_link": "end_effector_link",
                "limits": [(-math.pi, math.pi), (-1.5, 1.5), (-1.5, 1.4), 
                          (-1.7, 1.97), (-math.pi, math.pi)]
            },
            "5dof": {
                "joints": ["joint1", "joint2", "joint3", "joint4"],
                "end_link": "end_effector_link",
                "limits": [(-math.pi, math.pi), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97)]
            }
        }

    def load_poses(self, csv_file="generated_goals_5dof.csv"):
        """Load poses from Phase 1 CSV"""
        try:
            df = pd.read_csv(csv_file)
            required_cols = ["x", "y", "z", "qx", "qy", "qz", "qw"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain: {required_cols}")
            return df[required_cols].values.tolist()
        except Exception as e:
            self.get_logger().error(f"Failed to load {csv_file}: {e}")
            return []

    def get_start_state(self, config_name, start_mode, seed):
        """Generate start state"""
        np.random.seed(seed)
        config = self.configs[config_name]
        
        if start_mode == "home":
            return [0.0] * len(config["joints"])
        elif start_mode == "random":
            limits = config["limits"]
            return [np.random.uniform(lower, upper) for lower, upper in limits]
        else:
            raise ValueError(f"Invalid start_mode: {start_mode}")

    def compute_path_length(self, waypoints):
        """Compute joint-space path length: Σ||q(i+1)-q(i)||₂"""
        if len(waypoints) < 2:
            return 0.0
        
        path_length = 0.0
        for i in range(len(waypoints) - 1):
            q1 = np.array(waypoints[i])
            q2 = np.array(waypoints[i + 1])
            path_length += np.linalg.norm(q2 - q1)
        return float(path_length)

    def create_pose_constraints(self, pose, end_link, pos_tol, ori_tol):
        """Create position + orientation constraints from pose"""
        goal_constraints = Constraints()
        
        # Position constraint
        pos_const = PositionConstraint()
        pos_const.header.frame_id = "link1"
        pos_const.link_name = end_link
        
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [pos_tol]
        
        pos_const.constraint_region.primitives.append(sphere)
        pos_const.constraint_region.primitive_poses.append(pose)
        pos_const.weight = 1.0
        goal_constraints.position_constraints.append(pos_const)
        
        # Orientation constraint
        # ori_const = OrientationConstraint()
        # ori_const.header.frame_id = "link1"
        # ori_const.link_name = end_link
        # ori_const.orientation = pose.orientation
        # ori_const.absolute_x_axis_tolerance = ori_tol
        # ori_const.absolute_y_axis_tolerance = ori_tol
        # ori_const.absolute_z_axis_tolerance = ori_tol
        # ori_const.weight = 1.0
        # goal_constraints.orientation_constraints.append(ori_const)
        
        return goal_constraints

    def run_benchmark(self, config_name, poses, start_mode, args):
        """Run benchmark for single config"""
        config = self.configs[config_name]
        results = []
        np.random.seed(args.seed)
        
        self.get_logger().info(f"🏃 Running {config_name} benchmark ({len(poses)} trials)...")
        
        start_experiment = time.time()
        experiment_start_utc = datetime.now(timezone.utc).isoformat()
        
        for trial_idx, pose_list in enumerate(poses[:args.trials]):
            trial_start = time.time()
            
            # Create pose message
            from geometry_msgs.msg import PoseStamped, Pose
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "link1"
            pose_stamped.pose.position.x = pose_list[0]
            pose_stamped.pose.position.y = pose_list[1]
            pose_stamped.pose.position.z = pose_list[2]
            # pose_stamped.pose.orientation.x = pose_list[3]
            # pose_stamped.pose.orientation.y = pose_list[4]
            # pose_stamped.pose.orientation.z = pose_list[5]
            # pose_stamped.pose.orientation.w = pose_list[6]
            
            # Planning request
            req = GetMotionPlan.Request()
            req.motion_plan_request.group_name = "arm"
            req.motion_plan_request.num_planning_attempts = args.num_planning_attempts
            req.motion_plan_request.allowed_planning_time = args.planning_time
            
            # Start state
            start_state = self.get_start_state(config_name, start_mode, args.seed + trial_idx)
            req.motion_plan_request.start_state.joint_state.name = config["joints"]
            req.motion_plan_request.start_state.joint_state.position = start_state
            
            # Goal constraints
            goal_constraints = self.create_pose_constraints(
                pose_stamped.pose, config["end_link"], args.pos_tol, args.ori_tol
            )
            req.motion_plan_request.goal_constraints.append(goal_constraints)
            
            # Execute planning
            plan_start = time.time()
            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=args.planning_time + 1.0)
            wall_time = time.time() - plan_start
            
            # Process result
            success = False
            planning_time = wall_time
            path_length = 0.0
            trajectory_duration = 0.0
            waypoints = []
            
            if future.result() and future.result().motion_plan_response:
                resp = future.result().motion_plan_response
                success = (resp.error_code.val == resp.error_code.SUCCESS)
                if success:
                    planning_time = resp.planning_time if resp.planning_time > 0 else wall_time
                    if resp.trajectory.joint_trajectory.points:
                        waypoints = [[float(p.positions[i]) for i in range(len(config["joints"]))]
                                   for p in resp.trajectory.joint_trajectory.points]
                        path_length = self.compute_path_length(waypoints)
                        trajectory_duration = resp.trajectory.joint_trajectory.points[-1].time_from_start.sec + \
                                            resp.trajectory.joint_trajectory.points[-1].time_from_start.nanosec * 1e-9
            
            trial_end = time.time()
            elapsed_time = trial_end - start_experiment
            
            # Store results
            result = {
                "trial": trial_idx,
                "success": success,
                "planning_time": planning_time,
                "wall_time": wall_time,
                "path_length_joint": path_length,
                "trajectory_duration": trajectory_duration,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_time_sec": elapsed_time,
                "start_state_json": json.dumps(start_state),
                "goal_pose_json": json.dumps(pose_list),
                "waypoints_json": json.dumps(waypoints)
            }
            results.append(result)
            
            if (trial_idx + 1) % 1 == 0:
                success_rate = np.mean([r["success"] for r in results]) * 100
                self.get_logger().info(f"  {trial_idx+1}/{args.trials} - Success: {success_rate:.1f}%")
        return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Pose Benchmarking Script")
    parser.add_argument("--config", required=True, choices=["5dof", "6dof", "both"])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--planning_time", type=float, default=10.0)
    parser.add_argument("--pos_tol", type=float, default=0.003)
    parser.add_argument("--ori_tol", type=float, default=0.05)
    parser.add_argument("--start_mode", choices=["home", "random"], default="home")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_planning_attempts", type=int, default=2)
    
    args = parser.parse_args()
    
    rclpy.init()
    benchmark = PoseBenchmark()
    
    # Load poses from Phase 1
    poses = benchmark.load_poses()
    if not poses:
        print("❌ No poses loaded from generated_goals_5dof.csv")
        return
    
    results_dfs = {}
    
    # Run benchmarks based on config
    configs_to_run = [args.config] if args.config != "both" else ["6dof", "5dof"]
    
    for config in configs_to_run:
        print(f"\n{'='*60}")
        print(f"🏁 BENCHMARKING {config.upper()} ARM")
        print(f"{'='*60}")
        print(f"Poses: {min(args.trials, len(poses))}, Tol: {args.pos_tol}m/{args.ori_tol}rad")
        print(f"Start: {args.start_mode}, Time: {args.planning_time}s")
        
        df = benchmark.run_benchmark(config, poses, args.start_mode, args)
        results_dfs[config] = df
        
        # Save results
        filename = f"benchmark_{config}_results.csv"
        df.to_csv(filename, index=False)
        success_rate = df['success'].mean() * 100
        print(f"\n✅ {config.upper()} COMPLETE!")
        print(f"📊 Success Rate: {success_rate:.2f}%")
        print(f"💾 Saved: {filename}")
    
    # Combined results if both
    if args.config == "both" and len(results_dfs) == 2:
        combined_df = pd.concat([results_dfs["6dof"], results_dfs["5dof"]], 
                               ignore_index=True, keys=["6dof", "5dof"])
        combined_df.to_csv("benchmark_both_results.csv")
        print(f"\n🎉 COMBINED RESULTS: benchmark_both_results.csv")
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()
