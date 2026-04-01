#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration

import argparse
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timezone
import math
import glob
import os
import random

# --- CONFIGURABLE CONSTANTS ---
BASE_FRAME = "link1"
MOVE_GROUP_NAME = "arm" 

JOINT_LIMITS = {
    "joint1": (-math.pi, math.pi),
    "joint2": (-1.5, 1.5),
    "joint3": (-1.5, 1.4),
    "joint4": (-1.7, 1.97),
    "joint5_roll": (-math.pi, math.pi)
}

class MoveItBenchmarker(Node):
    def __init__(self):
        super().__init__('moveit2_advanced_benchmarker')
        self.get_logger().info("Initializing Native ROS 2 MoveIt Benchmarker...")
        
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.move_action = ActionClient(self, MoveGroup, '/move_action')
        
        self.get_logger().info("Waiting for /compute_ik service...")
        self.ik_client.wait_for_service()
        self.get_logger().info("Waiting for /move_action server...")
        self.move_action.wait_for_server()
        self.get_logger().info("✅ MoveIt 2 bindings established.")

    def get_start_joints(self, mode, joint_names):
        if mode == "home":
            return [0.0] * len(joint_names)
        else:
            return [float(np.random.uniform(JOINT_LIMITS[j][0], JOINT_LIMITS[j][1])) for j in joint_names]

    def compute_ik(self, x, y, z, joint_names):
        """RAPID-FIRE IK BRUTE-FORCING: Tries up to 40 random seeds to unstuck KDL"""
        req = GetPositionIK.Request()
        ik_req = PositionIKRequest()
        ik_req.group_name = MOVE_GROUP_NAME
        ik_req.avoid_collisions = True
        
        # Keep timeout extremely short per seed (0.02s) so we can loop 40 times fast
        ik_req.timeout = Duration(sec=0, nanosec=20000000) 
        
        pose = PoseStamped()
        pose.header.frame_id = BASE_FRAME
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.w = 1.0 # Ignored by KDL when position_only_ik is True
        ik_req.pose_stamped = pose
        
        # Test 40 different physical starting positions
        for attempt in range(40):
            if attempt == 0:
                seed = [0.0] * len(joint_names)  # Home
            elif attempt == 1:
                seed = [0.5] * len(joint_names)  # Bent forward
            elif attempt == 2:
                seed = [-0.5] * len(joint_names) # Bent backward
            else:
                seed = [float(np.random.uniform(JOINT_LIMITS[j][0], JOINT_LIMITS[j][1])) for j in joint_names]
                
            ik_req.robot_state.joint_state.name = joint_names
            ik_req.robot_state.joint_state.position = seed
            req.ik_request = ik_req
            
            future = self.ik_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()
            
            # Error Code 1 == SUCCESS
            if resp and resp.error_code.val == 1:
                solution_dict = dict(zip(resp.solution.joint_state.name, resp.solution.joint_state.position))
                try:
                    return [solution_dict[j] for j in joint_names]
                except KeyError:
                    return None
                    
        return None # Failed all 40 seeds

    def plan_to_joint_goals(self, start_joints, goal_joints, joint_names, args):
        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        
        req.group_name = MOVE_GROUP_NAME
        req.num_planning_attempts = args.num_planning_attempts
        req.allowed_planning_time = float(args.planning_time)
        req.max_velocity_scaling_factor = 1.0
        req.max_acceleration_scaling_factor = 1.0

        joint_state = JointState()
        joint_state.name = joint_names
        joint_state.position = start_joints
        req.start_state.joint_state = joint_state

        constraint = Constraints()
        for name, pos in zip(joint_names, goal_joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = float(args.pos_tol)
            jc.tolerance_below = float(args.pos_tol)
            jc.weight = 1.0
            constraint.joint_constraints.append(jc)
        
        req.goal_constraints.append(constraint)
        goal_msg.request = req

        wall_start = time.time()
        send_goal_future = self.move_action.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            return False, 0.0, 0.0, 0.0, []

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result_response = get_result_future.result()
        
        wall_time = time.time() - wall_start
        result = result_response.result
        
        if result.error_code.val == 1:
            traj = result.planned_trajectory.joint_trajectory
            planning_time = result.planning_time
            waypoints = [list(pt.positions) for pt in traj.points]
            
            path_length = 0.0
            for i in range(1, len(waypoints)):
                p1 = np.array(waypoints[i-1])
                p2 = np.array(waypoints[i])
                path_length += np.linalg.norm(p2 - p1)
                
            if len(traj.points) > 0:
                dur = traj.points[-1].time_from_start
                traj_duration = dur.sec + (dur.nanosec * 1e-9)
            else:
                traj_duration = 0.0
                
            return True, planning_time, wall_time, path_length, traj_duration, waypoints
            
        return False, 0.0, wall_time, 0.0, 0.0, []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, choices=["5dof", "6dof", "both"], required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--planning_time", type=float, default=2.0)
    parser.add_argument("--pos_tol", type=float, default=0.01)
    parser.add_argument("--start_mode", type=str, choices=["home", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_planning_attempts", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    rclpy.init()
    node = MoveItBenchmarker()

    configs_to_run = ["5dof", "6dof"] if args.config == "both" else [args.config]

    files = glob.glob("robot_efficiency_benchmark_*.csv")
    if not files:
        node.get_logger().error("No benchmark pose CSV found!")
        return
    csv_filename = max(files, key=os.path.getctime)
    df = pd.read_csv(csv_filename)
    poses = df[['x', 'y', 'z']].values.tolist()
    
    if len(poses) > args.trials:
        poses = poses[:args.trials]

    print(f"\n🚀 LOADED {len(poses)} TARGETS FROM: {csv_filename}")

    for config in configs_to_run:
        print(f"\n" + "="*60)
        print(f"🤖 STARTING BENCHMARK: {config.upper()} ARM")
        print("="*60)

        joint_names = ["joint1", "joint2", "joint3", "joint4"]
        if config == "6dof":
            joint_names.append("joint5_roll")

        main_log = []
        ik_log = []

        for trial, target in enumerate(poses):
            x, y, z = target[0], target[1], target[2]
            trial_start_time = time.time()
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # 1. Get Planning Start State
            start_joints = node.get_start_joints(args.start_mode, joint_names)
            
            # 2. Compute IK (Now uses internal safe seeds)
            ik_solution = node.compute_ik(x, y, z, joint_names)
            
            ik_success = (ik_solution is not None)
            
            ik_log.append({
                "trial": trial + 1, "x": x, "y": y, "z": z,
                "joint_angles_json": json.dumps(ik_solution) if ik_success else "[]",
                "success": ik_success
            })

            # 3. Plan to Joint Goals
            plan_success = False
            p_time, w_time, p_length, t_dur, waypoints = 0.0, 0.0, 0.0, 0.0, []
            
            if ik_success:
                plan_success, p_time, w_time, p_length, t_dur, waypoints = node.plan_to_joint_goals(
                    start_joints, ik_solution, joint_names, args
                )

            elapsed_sec = time.time() - trial_start_time

            main_log.append({
                "trial": trial + 1, "success": plan_success,
                "planning_time": p_time, "wall_time": w_time,
                "path_length_joint": p_length, "trajectory_duration": t_dur,
                "timestamp_utc": timestamp, "elapsed_time_sec": elapsed_sec,
                "start_state_json": json.dumps(start_joints),
                "goal_pose_json": json.dumps({"x": x, "y": y, "z": z}),
                "waypoints_json": json.dumps(waypoints)
            })

            status_emoji = "✅" if plan_success else ("❌ (Plan Fail)" if ik_success else "❌ (IK Fail)")
            print(f"[{trial+1:03d}/{len(poses)}] Target: ({x:6.3f}, {y:6.3f}, {z:6.3f}) -> {status_emoji}")

        main_df = pd.DataFrame(main_log)
        ik_df = pd.DataFrame(ik_log)

        main_out = f"benchmark_results_main_{config}_{args.trials}trials.csv"
        ik_out = f"benchmark_results_ik_{config}_{args.trials}trials.csv"

        main_df.to_csv(main_out, index=False)
        ik_df.to_csv(ik_out, index=False)
        
        success_rate = (main_df['success'].sum() / len(main_df)) * 100
        ik_success_rate = (ik_df['success'].sum() / len(ik_df)) * 100

        print("\n" + "📊 "*15)
        print(f"🏆 {config.upper()} BENCHMARK COMPLETE!")
        print(f"🎯 IK Success Rate:   {ik_success_rate:.1f}%")
        print(f"🎯 Plan Success Rate: {success_rate:.1f}%")
        print("📊 "*15)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient

# from moveit_msgs.srv import GetPositionIK
# from moveit_msgs.msg import PositionIKRequest
# from moveit_msgs.action import MoveGroup
# from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import PoseStamped

# import argparse
# import pandas as pd
# import numpy as np
# import json
# import time
# from datetime import datetime, timezone
# import math
# import glob
# import os
# import random

# # --- CONFIGURABLE CONSTANTS ---
# BASE_FRAME = "link1"
# END_EFFECTOR_LINK = "end_effector_link"
# MOVE_GROUP_NAME = "arm" # Change to "manipulator" if that's what your config uses

# JOINT_LIMITS = {
#     "joint1": (-math.pi, math.pi),
#     "joint2": (-1.5, 1.5),
#     "joint3": (-1.5, 1.4),
#     "joint4": (-1.7, 1.97),
#     "joint5_roll": (-math.pi, math.pi)
# }

# class MoveItBenchmarker(Node):
#     def __init__(self):
#         super().__init__('moveit2_advanced_benchmarker')
        
#         self.get_logger().info("Initializing Native ROS 2 MoveIt Benchmarker...")
        
#         # 1. Setup MoveIt Clients
#         self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
#         self.move_action = ActionClient(self, MoveGroup, '/move_action')
        
#         # Wait for them
#         self.get_logger().info("Waiting for /compute_ik service...")
#         self.ik_client.wait_for_service()
#         self.get_logger().info("Waiting for /move_action server...")
#         self.move_action.wait_for_server()
        
#         self.get_logger().info("✅ MoveIt 2 bindings established.")

#     def get_start_joints(self, mode, joint_names):
#         """Generates the start state based on limits."""
#         if mode == "home":
#             return [0.0] * len(joint_names)
#         else:
#             return [float(np.random.uniform(JOINT_LIMITS[j][0], JOINT_LIMITS[j][1])) for j in joint_names]

#     def compute_ik(self, x, y, z, start_joints, joint_names):
#         """CONSTRAINT A: Call IK service to get Joint Angles from XYZ."""
#         req = GetPositionIK.Request()
#         ik_req = PositionIKRequest()
#         ik_req.group_name = MOVE_GROUP_NAME
#         ik_req.avoid_collisions = True
        
#         # We start the IK solver from our generated start_joints
#         ik_req.robot_state.joint_state.name = joint_names
#         ik_req.robot_state.joint_state.position = start_joints
        
#         # Target Cartesian Pose (Position Only effectively since 5DOF limits orientation)
#         pose = PoseStamped()
#         pose.header.frame_id = BASE_FRAME
#         pose.pose.position.x = float(x)
#         pose.pose.position.y = float(y)
#         pose.pose.position.z = float(z)
#         pose.pose.orientation.w = 1.0 # Default quaternion
#         ik_req.pose_stamped = pose
        
#         req.ik_request = ik_req
        
#         # Sync call simulation inside ROS2
#         future = self.ik_client.call_async(req)
#         rclpy.spin_until_future_complete(self, future)
#         resp = future.result()
        
#         # MoveIt Error Code 1 == SUCCESS
#         if resp.error_code.val == 1:
#             # Extract just the joints we care about
#             full_names = resp.solution.joint_state.name
#             full_positions = resp.solution.joint_state.position
            
#             # Map back to our specific ordered joint list
#             solution_dict = dict(zip(full_names, full_positions))
#             try:
#                 target_joints = [solution_dict[j] for j in joint_names]
#                 return target_joints
#             except KeyError:
#                 self.get_logger().error("IK returned success but missing required joints!")
#                 return None
#         return None

#     def plan_to_joint_goals(self, start_joints, goal_joints, joint_names, args):
#         """CONSTRAINT B: Feed computed Joint Angles as Goals to MoveIt 2."""
#         goal_msg = MoveGroup.Goal()
#         req = MotionPlanRequest()
        
#         req.group_name = MOVE_GROUP_NAME
#         req.num_planning_attempts = args.num_planning_attempts
#         req.allowed_planning_time = float(args.planning_time)
#         req.max_velocity_scaling_factor = 1.0
#         req.max_acceleration_scaling_factor = 1.0

#         # Start State
#         joint_state = JointState()
#         joint_state.name = joint_names
#         joint_state.position = start_joints
#         req.start_state.joint_state = joint_state

#         # Goal Constraints (Joint Space)
#         constraint = Constraints()
#         for name, pos in zip(joint_names, goal_joints):
#             jc = JointConstraint()
#             jc.joint_name = name
#             jc.position = float(pos)
#             jc.tolerance_above = float(args.pos_tol) # Applied as Radian tolerance for joints
#             jc.tolerance_below = float(args.pos_tol)
#             jc.weight = 1.0
#             constraint.joint_constraints.append(jc)
        
#         req.goal_constraints.append(constraint)
#         goal_msg.request = req

#         # Send Action
#         wall_start = time.time()
#         send_goal_future = self.move_action.send_goal_async(goal_msg)
#         rclpy.spin_until_future_complete(self, send_goal_future)
#         goal_handle = send_goal_future.result()

#         if not goal_handle.accepted:
#             return False, 0.0, 0.0, 0.0, []

#         get_result_future = goal_handle.get_result_async()
#         rclpy.spin_until_future_complete(self, get_result_future)
#         result_response = get_result_future.result()
        
#         wall_time = time.time() - wall_start
#         result = result_response.result
        
#         # Error Code 1 == SUCCESS
#         if result.error_code.val == 1:
#             traj = result.planned_trajectory.joint_trajectory
            
#             # Extract Metrics
#             planning_time = result.planning_time
            
#             waypoints = [list(pt.positions) for pt in traj.points]
            
#             # Calculate path length (Sum of Euclidean distances in joint space)
#             path_length = 0.0
#             for i in range(1, len(waypoints)):
#                 p1 = np.array(waypoints[i-1])
#                 p2 = np.array(waypoints[i])
#                 path_length += np.linalg.norm(p2 - p1)
                
#             # Trajectory Duration
#             if len(traj.points) > 0:
#                 dur = traj.points[-1].time_from_start
#                 traj_duration = dur.sec + (dur.nanosec * 1e-9)
#             else:
#                 traj_duration = 0.0
                
#             return True, planning_time, wall_time, path_length, traj_duration, waypoints
            
#         return False, 0.0, wall_time, 0.0, 0.0, []

# def main():
#     parser = argparse.ArgumentParser(description="MoveIt 2 Joint-Space Benchmarking Tool")
#     parser.add_argument("--config", type=str, choices=["5dof", "6dof", "both"], required=True)
#     parser.add_argument("--trials", type=int, default=100)
#     parser.add_argument("--planning_time", type=float, default=2.0)
#     parser.add_argument("--pos_tol", type=float, default=0.01) # Joint tolerance in rads
#     parser.add_argument("--start_mode", type=str, choices=["home", "random"], default="random")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--num_planning_attempts", type=int, default=5)
#     args = parser.parse_args()

#     # Set seeds for reproducibility
#     random.seed(args.seed)
#     np.random.seed(args.seed)

#     rclpy.init()
#     node = MoveItBenchmarker()

#     # Determine Configs
#     configs_to_run = ["5dof", "6dof"] if args.config == "both" else [args.config]
#     if args.config == "both":
#         print("\n⚠️  WARNING: Running 'both' requires your active MoveIt instance to dynamically support both joint limits. Ensure MoveIt is configured correctly!")

#     # Find the Benchmark Targets
#     files = glob.glob("robot_efficiency_benchmark_*.csv")
#     if not files:
#         node.get_logger().error("No benchmark pose CSV found! Please generate one first.")
#         return
#     csv_filename = max(files, key=os.path.getctime)
#     df = pd.read_csv(csv_filename)
#     poses = df[['x', 'y', 'z']].values.tolist()
    
#     # Cap to requested trials
#     if len(poses) > args.trials:
#         poses = poses[:args.trials]

#     print(f"\n🚀 LOADED {len(poses)} TARGETS FROM: {csv_filename}")
#     print(f"⚙️  SETTINGS: Start={args.start_mode}, Time={args.planning_time}s, Attmps={args.num_planning_attempts}")

#     for config in configs_to_run:
#         print(f"\n" + "="*60)
#         print(f"🤖 STARTING BENCHMARK: {config.upper()} ARM")
#         print("="*60)

#         joint_names = ["joint1", "joint2", "joint3", "joint4"]
#         if config == "6dof":
#             joint_names.append("joint5_roll")

#         main_log = []
#         ik_log = []

#         for trial, target in enumerate(poses):
#             x, y, z = target[0], target[1], target[2]
            
#             # Timing overall process
#             trial_start_time = time.time()
#             timestamp = datetime.now(timezone.utc).isoformat()
            
#             # Step 1: Start State
#             start_joints = node.get_start_joints(args.start_mode, joint_names)
            
#             # Step 2: Compute IK
#             ik_solution = node.compute_ik(x, y, z, start_joints, joint_names)
            
#             ik_success = (ik_solution is not None)
            
#             # Log for Second CSV (Verification/IK Data)
#             ik_log.append({
#                 "trial": trial + 1,
#                 "x": x, "y": y, "z": z,
#                 "joint_angles_json": json.dumps(ik_solution) if ik_success else "[]",
#                 "success": ik_success
#             })

#             # Step 3: Plan via Joint Goals
#             plan_success = False
#             p_time, w_time, p_length, t_dur, waypoints = 0.0, 0.0, 0.0, 0.0, []
            
#             if ik_success:
#                 plan_success, p_time, w_time, p_length, t_dur, waypoints = node.plan_to_joint_goals(
#                     start_joints, ik_solution, joint_names, args
#                 )

#             elapsed_sec = time.time() - trial_start_time

#             # Log for Main CSV
#             main_log.append({
#                 "trial": trial + 1,
#                 "success": plan_success,
#                 "planning_time": p_time,
#                 "wall_time": w_time,
#                 "path_length_joint": p_length,
#                 "trajectory_duration": t_dur,
#                 "timestamp_utc": timestamp,
#                 "elapsed_time_sec": elapsed_sec,
#                 "start_state_json": json.dumps(start_joints),
#                 "goal_pose_json": json.dumps({"x": x, "y": y, "z": z}),
#                 "waypoints_json": json.dumps(waypoints)
#             })

#             # Realtime Output Tracker
#             status_emoji = "✅" if plan_success else ("❌ (Plan Fail)" if ik_success else "❌ (IK Fail)")
#             print(f"[{trial+1:03d}/{len(poses)}] Target: ({x:6.3f}, {y:6.3f}, {z:6.3f}) -> {status_emoji}")

#         # Save specific config CSVs
#         main_df = pd.DataFrame(main_log)
#         ik_df = pd.DataFrame(ik_log)

#         main_out = f"benchmark_results_main_{config}_{args.trials}trials.csv"
#         ik_out = f"benchmark_results_ik_{config}_{args.trials}trials.csv"

#         main_df.to_csv(main_out, index=False)
#         ik_df.to_csv(ik_out, index=False)
        
#         success_rate = (main_df['success'].sum() / len(main_df)) * 100
#         ik_success_rate = (ik_df['success'].sum() / len(ik_df)) * 100

#         print("\n" + "📊 "*15)
#         print(f"🏆 {config.upper()} BENCHMARK COMPLETE!")
#         print(f"🎯 IK Success Rate:   {ik_success_rate:.1f}%")
#         print(f"🎯 Plan Success Rate: {success_rate:.1f}%")
#         print(f"💾 Saved Main Log to: {main_out}")
#         print(f"💾 Saved IK Log to:   {ik_out}")
#         print("📊 "*15)

#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == "__main__":
#     main()