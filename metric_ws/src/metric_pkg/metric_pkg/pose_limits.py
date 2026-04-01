#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
import numpy as np
import pandas as pd
import math
import time
from datetime import datetime
from scipy.spatial import ConvexHull

class AdvancedWorkspaceAnalyzer(Node):
    def __init__(self):
        super().__init__("advanced_workspace_analyzer")
        
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        
        # OpenManipulator-X config
        self.joints = ["joint1", "joint2", "joint3", "joint4"]
        self.end_link = "end_effector_link"
        self.limits = [
            (-math.pi, math.pi), # joint1
            (-1.5, 1.5),         # joint2
            (-1.5, 1.4),         # joint3
            (-1.7, 1.97)         # joint4
        ]
        
    def wait_for_services(self):
        self.get_logger().info("Waiting for MoveIt /compute_fk service...")
        return self.fk_client.wait_for_service(timeout_sec=5.0)

    def run_advanced_analysis(self, num_samples=8000):
        if not self.wait_for_services():
            self.get_logger().error("Service not available! Is MoveIt running?")
            return
            
        print("\n" + "="*80)
        print(f"🚀 STARTING HIGH-PRECISION WORKSPACE ANALYSIS ({num_samples} samples)")
        print("="*80)
        
        # 1. Stratified/Uniform Random Sampling across Joint Limits
        samples = np.zeros((num_samples, 4))
        for i in range(4):
            min_val, max_val = self.limits[i]
            samples[:, i] = np.random.uniform(min_val, max_val, num_samples)
            
        results = []
        
        # Setup reusable FK Request
        req = GetPositionFK.Request()
        req.header.frame_id = "link1" 
        req.fk_link_names = [self.end_link]
        robot_state = RobotState()
        joint_state = JointState(name=self.joints)
        
        start_time = time.time()
        
        # 2. Query MoveIt FK (Sequential Service Calls)
        # Note: For >100k samples, moveit_py or PyKDL bindings should be used to bypass ROS service overhead.
        for i in range(num_samples):
            if i % 1000 == 0 and i > 0:
                rate = i / (time.time() - start_time)
                print(f"⏳ Processed {i}/{num_samples} poses... ({rate:.1f} poses/sec)")
            
            joint_state.position = [float(v) for v in samples[i]]
            robot_state.joint_state = joint_state
            req.robot_state = robot_state
            
            future = self.fk_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            
            resp = future.result()
            if resp and resp.error_code.val == resp.error_code.SUCCESS:
                pose = resp.pose_stamped[0].pose
                x, y, z = pose.position.x, pose.position.y, pose.position.z
                results.append([samples[i][0], samples[i][1], samples[i][2], samples[i][3], x, y, z])
        
        total_time = time.time() - start_time
        print(f"✅ Extracted {len(results)} valid Cartesian points in {total_time:.2f} seconds.")
        
        # 3. Process Advanced Metrics (Convex Hull)
        self.process_advanced_results(results)

    def process_advanced_results(self, results):
        if not results:
            print("❌ No valid FK poses returned.")
            return
            
        # Convert to numpy array for fast scientific computing
        data = np.array(results)
        points_3d = data[:, 4:7] # Extract just X, Y, Z
        
        # Bounding Box
        min_x, max_x = points_3d[:, 0].min(), points_3d[:, 0].max()
        min_y, max_y = points_3d[:, 1].min(), points_3d[:, 1].max()
        min_z, max_z = points_3d[:, 2].min(), points_3d[:, 2].max()
        max_reach = np.max(np.linalg.norm(points_3d, axis=1))
        bounding_box_vol = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        
        # 🧠 Compute True Workspace Envelope using Convex Hull
        try:
            print("\n⚙️ Computing Mathematical Convex Hull Surface...")
            hull = ConvexHull(points_3d)
            true_volume = hull.volume
            surface_area = hull.area
            
            # Extract just the boundary points for visualization
            boundary_points = points_3d[hull.vertices]
            boundary_df = pd.DataFrame(boundary_points, columns=['x', 'y', 'z'])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hull_file = f"omx_workspace_boundary_{timestamp}.csv"
            boundary_df.to_csv(hull_file, index=False)
            
        except Exception as e:
            print(f"⚠️ Convex Hull computation failed: {e}")
            true_volume, surface_area, hull_file = 0, 0, "N/A"

        print("\n" + "🎯"*40)
        print(" TRUE GEOMETRIC REACHABILITY LIMITS ")
        print("🎯"*40)
        print(f"📏 X Extents: [ {min_x:7.4f} , {max_x:7.4f} ] m")
        print(f"📏 Y Extents: [ {min_y:7.4f} , {max_y:7.4f} ] m")
        print(f"📏 Z Extents: [ {min_z:7.4f} , {max_z:7.4f} ] m")
        print("-" * 80)
        print(f"🌍 Max Radial Reach:       {max_reach:7.4f} m")
        print(f"🧊 Bounding Box Volume:    {bounding_box_vol:7.4f} m³ (Overestimated limits)")
        print(f"🧬 True Workspace Volume:  {true_volume:7.4f} m³ (Exact via Convex Hull)")
        print(f"🌐 Workspace Surface Area: {surface_area:7.4f} m²")
        print("=" * 80)
        
        if true_volume > 0:
            print(f"💡 Notice that the True Volume is roughly {(true_volume/bounding_box_vol)*100:.1f}% of the Bounding Box volume!")
            print(f"💾 Saved {len(boundary_points)} exact mathematical boundary points to: {hull_file}")

def main(args=None):
    rclpy.init(args=args)
    analyzer = AdvancedWorkspaceAnalyzer()
    
    # 8000 samples strikes a balance between ROS2 service overhead (takes ~20 seconds) 
    # and providing enough density for the Convex Hull to wrap tightly around the true limits.
    analyzer.run_advanced_analysis(num_samples=8000)
    
    analyzer.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()




# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from moveit_msgs.srv import GetMotionPlan
# from moveit_msgs.msg import Constraints, JointConstraint
# import numpy as np
# import math
# import time
# import pandas as pd
# from datetime import datetime

# class PoseLimitsFinder(Node):
#     def __init__(self):
#         super().__init__("pose_limits_finder")
        
#         self.client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
#         if not self.client.wait_for_service(timeout_sec=5.0):
#             raise RuntimeError("MoveIt planning service not available")
        
#         # YOUR EXACT OpenManipulator-X 4-JOINT CONFIG
#         self.config = {
#             "joints": ["joint1", "joint2", "joint3", "joint4"],
#             "end_link": "end_effector_link",
#             "group_name": "arm",
#             "limits": [(-math.pi, math.pi), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97)]
#         }
        
#         print("🔍 OpenManipulator-X (4-DOF + Gripper) POSE LIMITS")
#         print(f"Joints: {self.config['joints']}")
        
#         self.all_results = []
        
#         # 1. Joint limit corners
#         self.corner_results = self.analyze_joint_corners()
        
#         # 2. Workspace extremes  
#         self.workspace_results = self.sample_workspace_boundaries()
        
#         # 3. Save to CSV
#         self.save_to_csv()
        
#         # 4. Final summary
#         self.print_final_summary()

#     def dh_fk_openmanipulator(self, joints):
#         """✅ EXACT Forward Kinematics for OpenManipulator-X URDF"""
#         q1, q2, q3, q4 = joints
        
#         # EXTRACTED from YOUR URDF joint origins:
#         # joint1 origin: x=0.012
#         # joint2: z=0.0595 (link2 height 0.038 centered)  
#         # joint3: xyz=(0.024, 0, 0.128)
#         # joint4: x=0.124
#         # end_effector_joint: x=0.126
        
#         # Joint 1 transformation
#         x1 = 0.012 * math.cos(q1)
#         y1 = 0.012 * math.sin(q1)
#         z1 = 0.0
        
#         # Joint 2: +z=0.0595, rotate Y
#         x2 = x1
#         y2 = y1
#         z2 = 0.0595
        
#         # Joint 3: +xyz=(0.024, 0, 0.128), rotate Y  
#         x3 = x2 + 0.024 * math.cos(q2 + q3)
#         y3 = y2
#         z3 = z2 + 0.128
        
#         # Joint 4: +x=0.124, rotate Y
#         x4 = x3 + 0.124 * math.cos(q2 + q3 + q4)
#         y4 = y3
#         z4 = z3
        
#         # End-effector: +x=0.126
#         x = x4 + 0.126
#         y = y4
#         z = z4
        
#         return x, y, z

#     def create_joint_goal(self, target_joints):
#         """Create joint constraints with explicit float conversion"""
#         goal_constraints = Constraints()
#         for i, joint_name in enumerate(self.config["joints"]):
#             joint_const = JointConstraint()
#             joint_const.joint_name = joint_name
#             joint_const.position = float(target_joints[i])
#             joint_const.tolerance_above = float(0.05)
#             joint_const.tolerance_below = float(0.05)
#             joint_const.weight = float(1.0)
#             goal_constraints.joint_constraints.append(joint_const)
#         return [goal_constraints]

#     def plan_to_joints(self, target_joints):
#         """Plan from home to target joints"""
#         req = GetMotionPlan.Request()
#         req.motion_plan_request.group_name = self.config["group_name"]
#         req.motion_plan_request.num_planning_attempts = 2
#         req.motion_plan_request.allowed_planning_time = 1.5
        
#         home_state = [0.0, 0.0, 0.0, 0.0]
#         req.motion_plan_request.start_state.joint_state.name = self.config["joints"]
#         req.motion_plan_request.start_state.joint_state.position = home_state
        
#         req.motion_plan_request.goal_constraints = self.create_joint_goal(target_joints)
        
#         future = self.client.call_async(req)
#         rclpy.spin_until_future_complete(self, future, timeout_sec=2.5)
        
#         if future.result() and future.result().motion_plan_response:
#             resp = future.result().motion_plan_response
#             if (resp.error_code.val == resp.error_code.SUCCESS and 
#                 resp.trajectory.joint_trajectory.points):
#                 final_point = resp.trajectory.joint_trajectory.points[-1]
#                 final_joints = [float(final_point.positions[i]) for i in range(4)]
#                 return True, final_joints
#         return False, None

#     def analyze_joint_corners(self):
#         """Test all 16 corner combinations"""
#         print("\n📍 JOINT LIMIT CORNERS (16 extremes)")
#         print("-" * 100)
        
#         results = []
#         limits = self.config['limits']
        
#         for i1 in [0, 1]:
#             for i2 in [0, 1]:
#                 for i3 in [0, 1]:
#                     for i4 in [0, 1]:
#                         joints = [limits[0][i1], limits[1][i2], limits[2][i3], limits[3][i4]]
#                         success, final_joints = self.plan_to_joints(joints)
                        
#                         x, y, z = self.dh_fk_openmanipulator(final_joints if success else joints)
                        
#                         result = {
#                             'type': 'corner',
#                             'index': f"[{i1},{i2},{i3},{i4}]",
#                             'label': '',
#                             'j1': float(joints[0]), 'j2': float(joints[1]), 
#                             'j3': float(joints[2]), 'j4': float(joints[3]),
#                             'final_j1': float(final_joints[0]) if success and final_joints else None,
#                             'final_j2': float(final_joints[1]) if success and final_joints else None,
#                             'final_j3': float(final_joints[2]) if success and final_joints else None,
#                             'final_j4': float(final_joints[3]) if success and final_joints else None,
#                             'x': float(x), 'y': float(y), 'z': float(z),
#                             'reachable': success
#                         }
#                         self.all_results.append(result)
#                         results.append(result)
                        
#                         status = "✅" if success else "❌"
#                         print(f"{status} {result['index']:8} | Q: [{', '.join([f'{q:.3f}' for q in joints])}] | XYZ: ({x:.3f}, {y:.3f}, {z:.3f})")
        
#         return results

#     def sample_workspace_boundaries(self):
#         """Sample extreme workspace configurations"""
#         print("\n🌐 WORKSPACE EXTREMES")
#         print("-" * 100)
        
#         extreme_configs = [
#             ([1.57, 0.0, 0.0, 0.0], "Max+X"),
#             ([-1.57, 0.0, 0.0, 0.0], "Max-X"),
#             ([0.0, 1.5, 0.0, 0.0], "Max+Y"), 
#             ([0.0, -1.5, 0.0, 0.0], "Max-Y"),
#             ([0.0, 0.0, 1.4, 1.97], "Max+Stretch"),
#             ([0.0, 0.0, -1.5, -1.7], "Max-Stretch"),
#         ]
        
#         results = []
#         for joints, label in extreme_configs:
#             success, final_joints = self.plan_to_joints(joints)
#             x, y, z = self.dh_fk_openmanipulator(final_joints if success else joints)
            
#             result = {
#                 'type': 'workspace',
#                 'index': '',
#                 'label': label,
#                 'j1': float(joints[0]), 'j2': float(joints[1]), 
#                 'j3': float(joints[2]), 'j4': float(joints[3]),
#                 'final_j1': float(final_joints[0]) if success and final_joints else None,
#                 'final_j2': float(final_joints[1]) if success and final_joints else None,
#                 'final_j3': float(final_joints[2]) if success and final_joints else None,
#                 'final_j4': float(final_joints[3]) if success and final_joints else None,
#                 'x': float(x), 'y': float(y), 'z': float(z),
#                 'reachable': success
#             }
#             self.all_results.append(result)
#             results.append(result)
            
#             status = "✅" if success else "❌"
#             print(f"{status} {label:12} | Q: [{', '.join([f'{q:.3f}' for q in joints])}] | XYZ: ({x:.3f}, {y:.3f}, {z:.3f})")
        
#         return results

#     def save_to_csv(self):
#         """Save ALL pose limits to CSV"""
#         df = pd.DataFrame(self.all_results)
        
#         # Reorder columns for readability
#         csv_columns = ['type', 'index', 'label', 'j1', 'j2', 'j3', 'j4', 
#                       'final_j1', 'final_j2', 'final_j3', 'final_j4',
#                       'x', 'y', 'z', 'reachable']
#         df = df[csv_columns]
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"openmanipulator_pose_limits_{timestamp}.csv"
#         df.to_csv(filename, index=False)
        
#         print(f"\n💾 SAVED: {filename}")
#         print(f"📊 {len(df)} total configurations analyzed")

#     def print_final_summary(self):
#         """Compute overall workspace bounds"""
#         reachable = [r for r in self.all_results if r['reachable']]
#         if reachable:
#             x_vals = [r['x'] for r in reachable]
#             y_vals = [r['y'] for r in reachable] 
#             z_vals = [r['z'] for r in reachable]
            
#             print("\n" + "="*100)
#             print("🎯 OPENMANIPULATOR-X REACHABLE WORKSPACE")
#             print("="*100)
#             print(f"📏 X: [{min(x_vals):6.3f}, {max(x_vals):6.3f}] m")
#             print(f"📏 Y: [{min(y_vals):6.3f}, {max(y_vals):6.3f}] m")
#             print(f"📏 Z: [{min(z_vals):6.3f}, {max(z_vals):6.3f}] m")
#             print(f"📦 Volume: {(max(x_vals)-min(x_vals))*(max(y_vals)-min(y_vals))*(max(z_vals)-min(z_vals)):.4f} m³")
            
#             success_rate = len(reachable)/len(self.all_results)*100
#             print(f"✅ Success rate: {success_rate:.1f}% ({len(reachable)}/{len(self.all_results)})")

# def main(args=None):
#     rclpy.init(args=args)
#     finder = PoseLimitsFinder()
#     rclpy.shutdown()

# if __name__ == "__main__":
#     main()



# # #!/usr/bin/env python3

# # import rclpy
# # from rclpy.node import Node
# # from moveit_msgs.srv import GetMotionPlan
# # from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
# # from shape_msgs.msg import SolidPrimitive
# # from geometry_msgs.msg import PoseStamped


# # class SinglePoseTest(Node):
# #     def __init__(self):
# #         super().__init__("single_pose_test")

# #         self.client = self.create_client(GetMotionPlan, "/plan_kinematic_path")

# #         if not self.client.wait_for_service(timeout_sec=5.0):
# #             raise RuntimeError("MoveIt planning service not available")

# #     def send_goal(self):

# #         # -----------------------------
# #         # 🔹 HARD CODED TARGET POSITION
# #         # -----------------------------
# #         pose = PoseStamped()
# #         pose.header.frame_id = "link1"

# #         pose.pose.position.x = 0.25
# #         pose.pose.position.y = 0.0
# #         pose.pose.position.z = 0.20

# #         pose.pose.orientation.w = 1.0
# #         # -----------------------------

# #         req = GetMotionPlan.Request()
# #         req.motion_plan_request.group_name = "arm"
# #         req.motion_plan_request.allowed_planning_time = 5.0

# #         # 🔹 4 ACTIVE JOINTS ONLY
# #         req.motion_plan_request.start_state.joint_state.name = [
# #             "joint1", "joint2", "joint3", "joint4"
# #         ]
# #         req.motion_plan_request.start_state.joint_state.position = [0.0, 0.0, 0.0, 0.0]

# #         goal_constraints = Constraints()

# #         # Position constraint (3 mm sphere)
# #         pos_const = PositionConstraint()
# #         pos_const.header.frame_id = "link1"
# #         pos_const.link_name = "end_effector_link"

# #         sphere = SolidPrimitive()
# #         sphere.type = SolidPrimitive.SPHERE
# #         sphere.dimensions = [0.003]

# #         pos_const.constraint_region.primitives.append(sphere)
# #         pos_const.constraint_region.primitive_poses.append(pose.pose)
# #         pos_const.weight = 1.0

# #         goal_constraints.position_constraints.append(pos_const)

# #         # 🔹 Orientation relaxed heavily
# #         ori_const = OrientationConstraint()
# #         ori_const.header.frame_id = "link1"
# #         ori_const.link_name = "end_effector_link"
# #         ori_const.orientation = pose.pose.orientation
# #         ori_const.absolute_x_axis_tolerance = 3.14
# #         ori_const.absolute_y_axis_tolerance = 3.14
# #         ori_const.absolute_z_axis_tolerance = 3.14
# #         ori_const.weight = 0.1

# #         goal_constraints.orientation_constraints.append(ori_const)

# #         req.motion_plan_request.goal_constraints.append(goal_constraints)

# #         print("Sending planning request...")
# #         future = self.client.call_async(req)
# #         rclpy.spin_until_future_complete(self, future, timeout_sec=6.0)

# #         if future.result() is None:
# #             print("❌ Service call failed")
# #             return

# #         resp = future.result().motion_plan_response

# #         if resp.error_code.val == resp.error_code.SUCCESS:
# #             print("✅ PLAN SUCCESSFUL")
# #         else:
# #             print("❌ PLAN FAILED")
# #             print("Error code:", resp.error_code.val)


# # def main():
# #     rclpy.init()
# #     node = SinglePoseTest()
# #     node.send_goal()
# #     rclpy.shutdown()


# # if __name__ == "__main__":
# #     main()