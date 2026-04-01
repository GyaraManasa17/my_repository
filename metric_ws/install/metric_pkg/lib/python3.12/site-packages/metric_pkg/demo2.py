#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
import math
import argparse
import glob
import os
from datetime import datetime

class PoseGenerator(Node):
    def __init__(self, csv_prefix="openmanipulator_pose_limits"):
        super().__init__("pose_generator")
        
        # Auto-find latest CSV file
        csv_file = self.find_latest_csv(csv_prefix)
        if not csv_file:
            raise RuntimeError(f"No CSV file found matching '{csv_prefix}*.csv'")
        
        # Load workspace limits
        self.workspace_limits = self.load_workspace_limits(csv_file)
        
        print(f"✅ Found CSV: {csv_file}")
        print(f"📏 Workspace: X[{self.workspace_limits['x_min']:.3f}, {self.workspace_limits['x_max']:.3f}]")
        print(f"📏             Y[{self.workspace_limits['y_min']:.3f}, {self.workspace_limits['y_max']:.3f}]")
        print(f"📏             Z[{self.workspace_limits['z_min']:.3f}, {self.workspace_limits['z_max']:.3f}]")

    def find_latest_csv(self, prefix):
        files = glob.glob(f"{prefix}*.csv")
        if not files:
            return None
        return max(files, key=os.path.getctime)

    def load_workspace_limits(self, csv_file):
        df = pd.read_csv(csv_file)
        reachable = df[df['reachable'] == True]
        return {
            'x_min': float(reachable['x'].min()),
            'x_max': float(reachable['x'].max()),
            'y_min': float(reachable['y'].min()),
            'y_max': float(reachable['y'].max()),
            'z_min': float(reachable['z'].min()),
            'z_max': float(reachable['z'].max())
        }

    def generate_random_pose(self):
        """Generate random pose WITH orientation variation"""
        x = np.random.uniform(self.workspace_limits['x_min'], self.workspace_limits['x_max'])
        y = np.random.uniform(self.workspace_limits['y_min'], self.workspace_limits['y_max']) 
        z = np.random.uniform(self.workspace_limits['z_min'], self.workspace_limits['z_max'])
        
        # Random orientation (roll, pitch, yaw variation)
        roll = np.random.uniform(-0.5, 0.5)   # Small roll
        pitch = np.random.uniform(-0.3, 0.3)  # Small pitch  
        yaw = np.random.uniform(-1.0, 1.0)    # Larger yaw
        
        # Convert Euler to quaternion
        qx, qy, qz, qw = self.euler_to_quaternion(roll, pitch, yaw)
        
        return [x, y, z, qx, qy, qz, qw]

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        
        return qx, qy, qz, qw

    def numerical_ik_openmanipulator(self, target_pose):
        """Generate IK joint seed for 4-DOF (5DOF will use same poses)"""
        x_d, y_d, z_d = target_pose[0], target_pose[1], target_pose[2]
        
        # Simplified DH lengths for IK seed
        L1, L2, L3, L4 = 0.012, 0.060, 0.152, 0.250
        
        best_joints = None
        best_error = float('inf')
        
        # Dense grid search
        q1_range = np.linspace(-1.5, 1.5, 20)
        q2_range = np.linspace(-1.0, 1.0, 20)
        q3_range = np.linspace(-1.5, 1.0, 20)
        
        for q1 in q1_range:
            for q2 in q2_range:
                for q3 in q3_range:
                    x_fk = (L1 * math.cos(q1) + L2 * math.cos(q1+q2) + 
                           L3 * math.cos(q1+q2+q3) + L4 * math.cos(q1+q2+q3))
                    y_fk = (L1 * math.sin(q1) + L2 * math.sin(q1+q2) + 
                           L3 * math.sin(q1+q2+q3) + L4 * math.sin(q1+q2+q3))
                    
                    error = math.sqrt((x_fk - x_d)**2 + (y_fk - y_d)**2)
                    
                    if error < best_error:
                        best_error = error
                        best_joints = [q1, q2, q3, 0.0]
        
        # Clamp to 4-DOF limits
        limits_4dof = [(-math.pi, math.pi), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97)]
        clamped_joints = [np.clip(best_joints[i], limits_4dof[i][0], limits_4dof[i][1]) for i in range(4)]
        
        return clamped_joints, best_error

    def generate_test_poses(self, num_poses=100):
        """Generate N standardized test poses for 5DOF vs 6DOF comparison"""
        print(f"\n🔄 Generating {num_poses} standardized test poses...")
        
        test_poses = []
        
        for i in range(num_poses):
            # Generate pose + orientation
            pose = self.generate_random_pose()
            joint_seed_4dof, ik_error = self.numerical_ik_openmanipulator(pose[:3])
            
            # Store complete test case
            test_case = {
                'pose_id': i,
                'x': pose[0], 'y': pose[1], 'z': pose[2],
                'qx': pose[3], 'qy': pose[4], 'qz': pose[5], 'qw': pose[6],
                'joint_seed_j1': joint_seed_4dof[0],
                'joint_seed_j2': joint_seed_4dof[1],
                'joint_seed_j3': joint_seed_4dof[2],
                'joint_seed_j4': joint_seed_4dof[3],
                'joint_seed_j5': 0.0,  # For 6DOF (roll joint)
                'ik_error': ik_error,
                'pos_tol': 0.01,  # Standardized tolerance
                'ori_tol': 0.2    # Standardized tolerance
            }
            test_poses.append(test_case)
            
            if (i + 1) % 20 == 0:
                print(f"   Generated {i+1}/{num_poses} poses...")
        
        return test_poses

    def save_test_poses(self, test_poses, filename_prefix="standardized_test_poses"):
        """Save standardized poses for 5DOF/6DOF comparison"""
        df = pd.DataFrame(test_poses)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        
        # Reorder columns for clarity
        columns = ['pose_id', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw',
                  'joint_seed_j1', 'joint_seed_j2', 'joint_seed_j3', 'joint_seed_j4', 'joint_seed_j5',
                  'ik_error', 'pos_tol', 'ori_tol']
        df = df[columns]
        
        df.to_csv(filename, index=False)
        print(f"\n💾 SAVED: {filename}")
        print(f"📊 {len(test_poses)} standardized test poses ready for 5DOF vs 6DOF benchmarking!")
        
        # Summary stats
        print(f"\n📈 POSE STATISTICS:")
        print(f"   X range: [{df['x'].min():.3f}, {df['x'].max():.3f}]")
        print(f"   Y range: [{df['y'].min():.3f}, {df['y'].max():.3f}]")
        print(f"   Z range: [{df['z'].min():.3f}, {df['z'].max():.3f}]")
        print(f"   IK error: {df['ik_error'].mean():.3f} ± {df['ik_error'].std():.3f}")

def main():
    parser = argparse.ArgumentParser(description="Generate Standardized Test Poses")
    parser.add_argument("--csv-prefix", default="openmanipulator_pose_limits", 
                       help="Workspace limits CSV prefix")
    parser.add_argument("--num-poses", type=int, default=100, 
                       help="Number of test poses to generate")
    parser.add_argument("--output-prefix", default="standardized_test_poses", 
                       help="Output CSV filename prefix")
    
    args = parser.parse_args()
    
    rclpy.init()
    generator = PoseGenerator(args.csv_prefix)
    test_poses = generator.generate_test_poses(args.num_poses)
    generator.save_test_poses(test_poses, args.output_prefix)
    rclpy.shutdown()

if __name__ == "__main__":
    main()







# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from moveit_msgs.srv import GetMotionPlan
# from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, JointConstraint
# from shape_msgs.msg import SolidPrimitive
# import pandas as pd
# import numpy as np
# import math
# import time
# import argparse
# import glob
# import os
# from geometry_msgs.msg import PoseStamped
# from datetime import datetime

# class RandomPosePlanner(Node):
#     def __init__(self, csv_prefix="openmanipulator_pose_limits"):
#         super().__init__("random_pose_planner")
        
#         # Auto-find latest CSV file
#         csv_file = self.find_latest_csv(csv_prefix)
#         if not csv_file:
#             raise RuntimeError(f"No CSV file found matching '{csv_prefix}*.csv'")
        
#         # Load workspace limits
#         self.workspace_limits = self.load_workspace_limits(csv_file)
        
#         # MoveIt planning service
#         self.client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
#         if not self.client.wait_for_service(timeout_sec=5.0):
#             raise RuntimeError("MoveIt planning service not available")
        
#         self.config = {
#             "joints": ["joint1", "joint2", "joint3", "joint4"],
#             "end_link": "end_effector_link", 
#             "group_name": "arm"
#         }
        
#         print(f"✅ Found CSV: {csv_file}")
#         print(f"📏 Workspace: X[{self.workspace_limits['x_min']:.3f}, {self.workspace_limits['x_max']:.3f}]")
#         print(f"📏             Y[{self.workspace_limits['y_min']:.3f}, {self.workspace_limits['y_max']:.3f}]")
#         print(f"📏             Z[{self.workspace_limits['z_min']:.3f}, {self.workspace_limits['z_max']:.3f}]")
        
#         self.results = []

#     def find_latest_csv(self, prefix):
#         """Auto-find latest openmanipulator_pose_limits_*.csv"""
#         files = glob.glob(f"{prefix}*.csv")
#         if not files:
#             return None
#         return max(files, key=os.path.getctime)

#     def load_workspace_limits(self, csv_file):
#         """Load CSV and extract min/max X,Y,Z from reachable poses"""
#         df = pd.read_csv(csv_file)
#         reachable = df[df['reachable'] == True]
        
#         if len(reachable) == 0:
#             raise ValueError("No reachable poses found in CSV!")
        
#         return {
#             'x_min': float(reachable['x'].min()),
#             'x_max': float(reachable['x'].max()),
#             'y_min': float(reachable['y'].min()),
#             'y_max': float(reachable['y'].max()),
#             'z_min': float(reachable['z'].min()),
#             'z_max': float(reachable['z'].max())
#         }

#     # ... [rest of the methods remain the same as previous script] ...

#     def generate_random_pose(self):
#         """Generate random pose within workspace limits"""
#         x = np.random.uniform(self.workspace_limits['x_min'], self.workspace_limits['x_max'])
#         y = np.random.uniform(self.workspace_limits['y_min'], self.workspace_limits['y_max']) 
#         z = np.random.uniform(self.workspace_limits['z_min'], self.workspace_limits['z_max'])
#         qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
#         return [x, y, z, qx, qy, qz, qw]

#     # [Include all other methods from previous script: numerical_ik_openmanipulator, 
#     #  create_pose_goal, pose_to_pose_msg, plan_to_pose, run_trials]

# def main():
#     parser = argparse.ArgumentParser(description="Random Pose Planner")
#     parser.add_argument("--csv-prefix", default="openmanipulator_pose_limits", 
#                        help="CSV prefix (default: openmanipulator_pose_limits)")
#     parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    
#     args = parser.parse_args()
    
#     rclpy.init()
#     planner = RandomPosePlanner(args.csv_prefix)
#     planner.run_trials(args.trials)
#     rclpy.shutdown()

# if __name__ == "__main__":
#     main()





# # #!/usr/bin/env python3

# # import rclpy
# # from rclpy.node import Node
# # from moveit_msgs.srv import GetMotionPlan
# # from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, JointConstraint
# # from shape_msgs.msg import SolidPrimitive
# # import pandas as pd
# # import numpy as np
# # import math
# # import time
# # import argparse
# # from geometry_msgs.msg import PoseStamped
# # from datetime import datetime

# # class RandomPosePlanner(Node):
# #     def __init__(self, csv_file):
# #         super().__init__("random_pose_planner")
        
# #         # Load workspace limits from your CSV
# #         self.workspace_limits = self.load_workspace_limits(csv_file)
        
# #         # MoveIt planning service (same as your benchmark)
# #         self.client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
# #         if not self.client.wait_for_service(timeout_sec=5.0):
# #             raise RuntimeError("MoveIt planning service not available")
        
# #         # OpenManipulator-X config
# #         self.config = {
# #             "joints": ["joint1", "joint2", "joint3", "joint4"],
# #             "end_link": "end_effector_link", 
# #             "group_name": "arm"
# #         }
        
# #         print(f"✅ Loaded workspace: X[{self.workspace_limits['x_min']:.3f}, {self.workspace_limits['x_max']:.3f}]")
# #         print(f"   Y[{self.workspace_limits['y_min']:.3f}, {self.workspace_limits['y_max']:.3f}]")
# #         print(f"   Z[{self.workspace_limits['z_min']:.3f}, {self.workspace_limits['z_max']:.3f}]")
        
# #         self.results = []

# #     def load_workspace_limits(self, csv_file):
# #         """Load CSV and extract min/max X,Y,Z from reachable poses"""
# #         df = pd.read_csv(csv_file)
# #         reachable = df[df['reachable'] == True]
        
# #         return {
# #             'x_min': float(reachable['x'].min()),
# #             'x_max': float(reachable['x'].max()),
# #             'y_min': float(reachable['y'].min()),
# #             'y_max': float(reachable['y'].max()),
# #             'z_min': float(reachable['z'].min()),
# #             'z_max': float(reachable['z'].max())
# #         }

# #     def generate_random_pose(self):
# #         """Generate random pose within workspace limits"""
# #         x = np.random.uniform(self.workspace_limits['x_min'], self.workspace_limits['x_max'])
# #         y = np.random.uniform(self.workspace_limits['y_min'], self.workspace_limits['y_max']) 
# #         z = np.random.uniform(self.workspace_limits['z_min'], self.workspace_limits['z_max'])
        
# #         # Simple orientation (identity quaternion)
# #         qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        
# #         return [x, y, z, qx, qy, qz, qw]

# #     def numerical_ik_openmanipulator(self, target_pose):
# #         """Simplified numerical IK for OpenManipulator-X"""
# #         x_d, y_d, z_d = target_pose[0], target_pose[1], target_pose[2]
        
# #         # DH params from your URDF
# #         L1, L2, L3, L4 = 0.012, 0.0595, 0.128, 0.124 + 0.126  # Total EE distance
        
# #         best_joints = None
# #         best_error = float('inf')
        
# #         # Grid search over reasonable joint ranges (faster than full optimization)
# #         q1_range = np.linspace(-1.0, 1.0, 20)
# #         q2_range = np.linspace(-1.0, 1.0, 20)
# #         q3_range = np.linspace(-1.0, 1.0, 20)
        
# #         for q1 in q1_range:
# #             for q2 in q2_range:
# #                 for q3 in q3_range:
# #                     # Forward kinematics
# #                     x_fk = (L1 * math.cos(q1) + 
# #                            L2 * math.cos(q1 + q2) + 
# #                            L3 * math.cos(q1 + q2 + q3) + 
# #                            L4 * math.cos(q1 + q2 + q3))
                    
# #                     y_fk = (L1 * math.sin(q1) + 
# #                            L2 * math.sin(q1 + q2) + 
# #                            L3 * math.sin(q1 + q2 + q3) + 
# #                            L4 * math.sin(q1 + q2 + q3))
                    
# #                     z_fk = 0.208  # Fixed Z from workspace
                    
# #                     error = math.sqrt((x_fk - x_d)**2 + (y_fk - y_d)**2)
                    
# #                     if error < best_error:
# #                         best_error = error
# #                         best_joints = [q1, q2, q3, 0.0]  # q4=0 for simplicity
        
# #         # Clamp to joint limits
# #         limits = [(-math.pi, math.pi), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97)]
# #         clamped_joints = [np.clip(best_joints[i], limits[i][0], limits[i][1]) for i in range(4)]
        
# #         return clamped_joints, best_error

# #     def create_pose_goal(self, pose_list, pos_tol=0.01, ori_tol=0.2):
# #         """Create pose constraints (same as your benchmark)"""
# #         goal_constraints = Constraints()
        
# #         # Position constraint
# #         pos_const = PositionConstraint()
# #         pos_const.header.frame_id = "link1"
# #         pos_const.link_name = self.config["end_link"]
        
# #         sphere = SolidPrimitive()
# #         sphere.type = SolidPrimitive.SPHERE
# #         sphere.dimensions = [pos_tol]
# #         pos_const.constraint_region.primitives.append(sphere)
# #         pos_const.constraint_region.primitive_poses.append(self.pose_to_pose_msg(pose_list))
# #         pos_const.weight = 1.0
# #         goal_constraints.position_constraints.append(pos_const)
        
# #         # Orientation constraint (loose)
# #         ori_const = OrientationConstraint()
# #         ori_const.header.frame_id = "link1"
# #         ori_const.link_name = self.config["end_link"]
# #         ori_const.orientation = self.pose_to_pose_msg(pose_list).orientation
# #         ori_const.absolute_x_axis_tolerance = ori_tol
# #         ori_const.absolute_y_axis_tolerance = ori_tol
# #         ori_const.absolute_z_axis_tolerance = ori_tol
# #         ori_const.weight = 0.5
# #         goal_constraints.orientation_constraints.append(ori_const)
        
# #         return [goal_constraints]

# #     def pose_to_pose_msg(self, pose_list):
# #         """Convert [x,y,z,qx,qy,qz,qw] to PoseStamped"""
# #         pose_stamped = PoseStamped()
# #         pose_stamped.header.frame_id = "link1"
# #         pose_stamped.pose.position.x = float(pose_list[0])
# #         pose_stamped.pose.position.y = float(pose_list[1])
# #         pose_stamped.pose.position.z = float(pose_list[2])
# #         pose_stamped.pose.orientation.x = float(pose_list[3])
# #         pose_stamped.pose.orientation.y = float(pose_list[4])
# #         pose_stamped.pose.orientation.z = float(pose_list[5])
# #         pose_stamped.pose.orientation.w = float(pose_list[6])
# #         return pose_stamped.pose

# #     def plan_to_pose(self, pose_list, joint_goals=None):
# #         """Plan to random pose using MoveIt2"""
# #         req = GetMotionPlan.Request()
# #         req.motion_plan_request.group_name = self.config["group_name"]
# #         req.motion_plan_request.num_planning_attempts = 5
# #         req.motion_plan_request.allowed_planning_time = 5.0
        
# #         # Start from home
# #         home_state = [0.0, 0.0, 0.0, 0.0]
# #         req.motion_plan_request.start_state.joint_state.name = self.config["joints"]
# #         req.motion_plan_request.start_state.joint_state.position = home_state
        
# #         # Pose goal
# #         goal_constraints = self.create_pose_goal(pose_list)
# #         req.motion_plan_request.goal_constraints = goal_constraints
        
# #         # Add joint goal seed if available
# #         if joint_goals is not None:
# #             joint_const_group = Constraints()
# #             for i, joint_name in enumerate(self.config["joints"]):
# #                 joint_const = JointConstraint()
# #                 joint_const.joint_name = joint_name
# #                 joint_const.position = float(joint_goals[i])
# #                 joint_const.tolerance_above = 0.3
# #                 joint_const.tolerance_below = 0.3
# #                 joint_const.weight = 0.8
# #                 joint_const_group.joint_constraints.append(joint_const)
# #             req.motion_plan_request.goal_constraints.append(joint_const_group)
        
# #         future = self.client.call_async(req)
# #         rclpy.spin_until_future_complete(self, future, timeout_sec=6.0)
        
# #         success = False
# #         planning_time = 0.0
# #         path_length = 0.0
        
# #         if future.result() and future.result().motion_plan_response:
# #             resp = future.result().motion_plan_response
# #             success = (resp.error_code.val == resp.error_code.SUCCESS)
# #             if success and resp.trajectory.joint_trajectory.points:
# #                 planning_time = resp.planning_time
# #                 waypoints = [[p.positions[i] for i in range(4)] 
# #                            for p in resp.trajectory.joint_trajectory.points]
# #                 path_length = sum(np.linalg.norm(np.array(waypoints[i+1]) - np.array(waypoints[i]))
# #                                 for i in range(len(waypoints)-1))
        
# #         return success, planning_time, path_length

# #     def run_trials(self, num_trials=50):
# #         """Run N random pose planning trials"""
# #         print(f"\n🚀 Running {num_trials} random pose trials...")
        
# #         for trial in range(num_trials):
# #             # 1. Generate random pose in workspace
# #             pose = self.generate_random_pose()
            
# #             # 2. Solve IK for joint seed
# #             joint_goals, ik_error = self.numerical_ik_openmanipulator(pose[:3])
            
# #             # 3. Plan to pose
# #             success, plan_time, path_len = self.plan_to_pose(pose, joint_goals)
            
# #             # 4. Store result
# #             result = {
# #                 'trial': trial,
# #                 'target_x': pose[0], 'target_y': pose[1], 'target_z': pose[2],
# #                 'ik_j1': joint_goals[0], 'ik_j2': joint_goals[1], 
# #                 'ik_j3': joint_goals[2], 'ik_j4': joint_goals[3],
# #                 'ik_error': ik_error,
# #                 'success': success,
# #                 'planning_time': plan_time,
# #                 'path_length': path_len
# #             }
# #             self.results.append(result)
            
# #             status = "✅" if success else "❌"
# #             print(f"{status} Trial {trial+1:2d} | Pose: ({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}) | "
# #                   f"IK_err: {ik_error:.3f} | Time: {plan_time:.2f}s")
            
# #             time.sleep(0.1)
        
# #         # Save results
# #         df = pd.DataFrame(self.results)
# #         df.to_csv(f"random_pose_planning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
# #                  index=False)
# #         print(f"\n💾 Results saved: random_pose_planning_results_*.csv")

# # def main():
# #     parser = argparse.ArgumentParser(description="Random Pose Planner from Workspace CSV")
# #     parser.add_argument("csv_file", help="Path to pose limits CSV")
# #     parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    
# #     args = parser.parse_args()
    
# #     rclpy.init()
# #     planner = RandomPosePlanner(args.csv_file)
# #     planner.run_trials(args.trials)
# #     rclpy.shutdown()

# # if __name__ == "__main__":
# #     main()
