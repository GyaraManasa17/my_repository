#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK, GetStateValidity
from moveit_msgs.msg import RobotState
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

class RealRobotHardwareValidator(Node):
    def __init__(self, csv_file):
        super().__init__('real_robot_hardware_validator')
        
        print("\n" + "="*80)
        print(" 🏆 PUBLICATION-GRADE HARDWARE VALIDATION FRAMEWORK")
        print("="*80)
        
        self.declare_parameter('controller_name', '/arm_controller/follow_joint_trajectory')
        self.action_topic = self.get_parameter('controller_name').get_parameter_value().string_value
        
        self.csv_file = csv_file
        self.NUM_REPEATS = 3  
        
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"[INFO] Loaded {len(self.df)} strategic test poses.")
            print(f"[INFO] Repeatability Mode: {self.NUM_REPEATS} iterations per pose.")
        except FileNotFoundError:
            self.get_logger().error(f"Cannot find {self.csv_file}.")
            exit()

        self.trajectory_client = ActionClient(self, FollowJointTrajectory, self.action_topic)
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        self.state_validity_client = self.create_client(GetStateValidity, "/check_state_validity")
        
        self.actual_joints =[0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.joint_names =['joint1', 'joint2', 'joint3', 'joint4', 'joint5_roll']
        self.end_link = "end_effector_link"
        self.HOME_POSE =[0.0, -1.0, 0.3, 0.7, 0.0]
        
        # ✅ FIX 2: Increased Jump threshold to 2.5 rad to allow full workspace reach from HOME
        self.MAX_SAFE_JUMP_RAD = 2.5  
        
        self.joint_limits =[
            (-3.14, 3.14),  # joint1
            (-1.5, 1.5),    # joint2
            (-1.5, 1.4),    # joint3
            (-1.7, 1.97),   # joint4
            (-3.0, 3.0)     # joint5
        ]
        
        self.execution_results =[]

    def joint_state_callback(self, msg):
        try:
            for i, name in enumerate(self.joint_names):
                if name in msg.name:
                    idx = msg.name.index(name)
                    self.actual_joints[i] = msg.position[idx]
        except ValueError:
            pass

    def wait_for_systems(self):
        print(f"[INFO] Waiting for Action Server ({self.action_topic})...")
        self.trajectory_client.wait_for_server()
        print(f"[INFO] Waiting for FK Service (/compute_fk)...")
        self.fk_client.wait_for_service()
        print(f"[INFO] Waiting for State Validity Service (/check_state_validity)...")
        self.state_validity_client.wait_for_service()
        print("[INFO] All Systems GO.\n")

    def passes_safety_check(self, row):
        joints = np.array([row['j1'], row['j2'], row['j3'], row['j4'], row['j5']])
        
        for i, j_val in enumerate(joints):
            min_lim, max_lim = self.joint_limits[i]
            if not (min_lim <= j_val <= max_lim):
                return False, 'safety_block_limits'
                
        if abs(row['pitch']) > 2.8:
            return False, 'safety_block_orientation'
            
        max_jump = np.max(np.abs(joints - np.array(self.HOME_POSE)))
        if max_jump > self.MAX_SAFE_JUMP_RAD:
            return False, 'safety_block_jump'
            
        sv_req = GetStateValidity.Request()
        sv_req.group_name = "arm"
        rs = RobotState()
        js = JointState(name=self.joint_names, position=joints.tolist())
        rs.joint_state = js
        sv_req.robot_state = rs
        
        sv_future = self.state_validity_client.call_async(sv_req)
        rclpy.spin_until_future_complete(self, sv_future)
        sv_resp = sv_future.result()

        if not sv_resp or not sv_resp.valid:
            return False, 'safety_block_collision'
            
        return True, 'safe'

    def compute_actual_pose(self, joints_array):
        req = GetPositionFK.Request()
        req.header.frame_id = "link1" 
        req.fk_link_names = [self.end_link]
        
        rs = RobotState()
        js = JointState(name=self.joint_names, position=joints_array)
        rs.joint_state = js
        req.robot_state = rs
        
        future = self.fk_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        
        if resp and resp.error_code.val == resp.error_code.SUCCESS:
            p = resp.pose_stamped[0].pose
            pos = np.array([p.position.x, p.position.y, p.position.z])
            quat = np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
            return pos, quat
        return None, None

    def execute_trajectory(self, target_joints, duration_sec=4.5):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = target_joints
        
        # ✅ FIX 1: Final target velocities must be strictly 0.0 for ROS trajectory controllers
        point.velocities = [0.0] * 5
        point.accelerations = [0.0] * 5
        
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
        goal_msg.trajectory.points = [point]

        send_goal_future = self.trajectory_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            return False

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result()
        
        return result.result.error_code == 0

    def run_experiment(self):
        self.wait_for_systems()
        total_poses = len(self.df)
        
        try:
            for index, row in self.df.iterrows():
                print(f"\n[{index+1}/{total_poses}] Testing: {row['test_type']}")
                
                is_safe, failure_reason = self.passes_safety_check(row)
                if not is_safe:
                    print(f"   ⚠️ Skipping entire pose. Reason: {failure_reason}")
                    for i in range(self.NUM_REPEATS):
                        self.log_data(index, row, i+1, 'failed', failure_reason)
                    continue

                target_joints =[row['j1'], row['j2'], row['j3'], row['j4'], row['j5']]
                
                for iteration in range(self.NUM_REPEATS):
                    self.execute_trajectory(self.HOME_POSE, duration_sec=3.0)
                    time.sleep(0.5)

                    success = self.execute_trajectory(target_joints, duration_sec=4.5)
                    
                    if not success:
                        print(f"   ❌ Hardware rejected trajectory on Iter {iteration+1}.")
                        self.log_data(index, row, iteration+1, 'failed', 'hardware_rejected')
                        continue
                    
                    time.sleep(2.0)
                    for _ in range(50):
                        rclpy.spin_once(self, timeout_sec=0.05)
                        
                    actual_j = self.actual_joints.copy()
                    actual_xyz, actual_q = self.compute_actual_pose(actual_j)
                    
                    if actual_xyz is None:
                        print(f"   ❌ FK failed to compute actual pose on Iter {iteration+1}.")
                        self.log_data(index, row, iteration+1, 'failed', 'fk_failed')
                        continue
                    
                    target_xyz = np.array([row['x'], row['y'], row['z']])
                    target_q = np.array([row['qx'], row['qy'], row['qz'], row['qw']])

                    cartesian_error_mm = np.linalg.norm(target_xyz - actual_xyz) * 1000.0
                    mean_j_error = np.mean(np.abs(np.array(target_joints) - np.array(actual_j)))
                    
                    dot = np.clip(np.abs(np.dot(target_q, actual_q)), 0.0, 1.0)
                    angle_error_deg = np.degrees(2 * np.arccos(dot))
                    
                    print(f"   🔄 Iter {iteration+1} | Pos Err: {cartesian_error_mm:8.6f} mm | Ang Err: {angle_error_deg:8.6f}°")
                    
                    self.log_data(index, row, iteration+1, 'success', 'none', actual_j, actual_xyz, actual_q, 
                                  cartesian_error_mm, angle_error_deg, mean_j_error)

            print("\n[INFO] Experiment Complete. Returning to HOME...")
            self.execute_trajectory(self.HOME_POSE, duration_sec=3.0)

        except KeyboardInterrupt:
            print("\n🚨 EMERGENCY STOP TRIGGERED! Returning to HOME safely...")
            self.execute_trajectory(self.HOME_POSE, duration_sec=3.0)
        
        finally:
            self.save_results()

    def log_data(self, index, row, iteration, status, failure_type, 
                 actual_j=None, actual_xyz=None, actual_q=None, 
                 cartesian_err=np.nan, angular_err=np.nan, joint_err=np.nan):
        
        self.execution_results.append({
            'pose_id': index,
            'iteration': iteration,
            'test_type': row['test_type'],
            'status': status,
            'failure_type': failure_type,
            
            'target_x': row['x'], 'target_y': row['y'], 'target_z': row['z'],
            'target_qx': row['qx'], 'target_qy': row['qy'], 'target_qz': row['qz'], 'target_qw': row['qw'],
            'cmd_j1': row['j1'], 'cmd_j2': row['j2'], 'cmd_j3': row['j3'], 'cmd_j4': row['j4'], 'cmd_j5': row['j5'],
            
            'actual_x': actual_xyz[0] if actual_xyz is not None else np.nan,
            'actual_y': actual_xyz[1] if actual_xyz is not None else np.nan,
            'actual_z': actual_xyz[2] if actual_xyz is not None else np.nan,
            'actual_qx': actual_q[0] if actual_q is not None else np.nan,
            'actual_qy': actual_q[1] if actual_q is not None else np.nan,
            'actual_qz': actual_q[2] if actual_q is not None else np.nan,
            'actual_qw': actual_q[3] if actual_q is not None else np.nan,
            'actual_j1': actual_j[0] if actual_j is not None else np.nan,
            'actual_j2': actual_j[1] if actual_j is not None else np.nan,
            'actual_j3': actual_j[2] if actual_j is not None else np.nan,
            'actual_j4': actual_j[3] if actual_j is not None else np.nan,
            'actual_j5': actual_j[4] if actual_j is not None else np.nan,

            'cartesian_error_mm': cartesian_err,
            'orientation_error_deg': angular_err,
            'mean_joint_error_rad': joint_err
        })

    def save_results(self):
        if len(self.execution_results) == 0:
            print("❌ No successful executions to save.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"hardware_validation_raw_data_{timestamp}.csv"
        txt_file = f"hardware_validation_metrics_{timestamp}.txt"
        
        results_df = pd.DataFrame(self.execution_results)
        results_df.to_csv(csv_file, index=False)
        
        success_df = results_df[results_df['status'] == 'success']
        failed_df = results_df[results_df['status'] == 'failed']
        
        total_attempts = len(results_df)
        success_rate = (len(success_df) / total_attempts) * 100 if total_attempts > 0 else 0
        
        avg_cartesian = success_df['cartesian_error_mm'].mean()
        max_cartesian = success_df['cartesian_error_mm'].max()
        min_cartesian = success_df['cartesian_error_mm'].min()
        
        avg_angular = success_df['orientation_error_deg'].mean()
        max_angular = success_df['orientation_error_deg'].max()
        
        grouped = success_df.groupby('pose_id')['cartesian_error_mm']
        std_dev_mean = grouped.std().mean()
        std_dev_max = grouped.std().max()
        
        fail_limits = len(failed_df[failed_df['failure_type'] == 'safety_block_limits'])
        fail_orient = len(failed_df[failed_df['failure_type'] == 'safety_block_orientation'])
        fail_jump = len(failed_df[failed_df['failure_type'] == 'safety_block_jump'])
        fail_collision = len(failed_df[failed_df['failure_type'] == 'safety_block_collision'])
        fail_hw = len(failed_df[failed_df['failure_type'] == 'hardware_rejected'])
        fail_fk = len(failed_df[failed_df['failure_type'] == 'fk_failed'])

        summary_text = (
            f"============================================================\n"
            f" 🎯 REAL-ROBOT HARDWARE VALIDATION & KINEMATIC METRICS\n"
            f"============================================================\n"
            f"Total Test Poses:           {len(self.df)}\n"
            f"Iterations per Pose:        {self.NUM_REPEATS}\n"
            f"Total Execution Attempts:   {total_attempts}\n"
            f"Successful Executions:      {len(success_df)}\n"
            f"Hardware Success Rate:      {success_rate:.2f}%\n"
            f"------------------------------------------------------------\n"
            f"📏 CARTESIAN POSITIONING (Accuracy)\n"
            f"------------------------------------------------------------\n"
            f"Mean Euclidean Error:       {avg_cartesian:7.4f} mm\n"
            f"Max Euclidean Error:        {max_cartesian:7.4f} mm\n"
            f"Min Euclidean Error:        {min_cartesian:7.4f} mm\n"
            f"------------------------------------------------------------\n"
            f"🔄 REPEATABILITY (Precision)\n"
            f"------------------------------------------------------------\n"
            f"Mean Spatial Std Dev:       ±{std_dev_mean:6.4f} mm\n"
            f"Max Spatial Std Dev:        ±{std_dev_max:6.4f} mm\n"
            f"------------------------------------------------------------\n"
            f"📐 ORIENTATION FIDELITY\n"
            f"------------------------------------------------------------\n"
            f"Mean Angular Error:         {avg_angular:7.4f}°\n"
            f"Max Angular Error:          {max_angular:7.4f}°\n"
            f"------------------------------------------------------------\n"
            f"⚠️ FAILURE ANALYSIS\n"
            f"------------------------------------------------------------\n"
            f"Total Failures:             {len(failed_df)}\n"
            f"Safety Blocks (Limits):     {fail_limits}\n"
            f"Safety Blocks (Orient/Jump):{fail_orient + fail_jump}\n"
            f"Safety Blocks (Collision):  {fail_collision}\n"
            f"Hardware Rejections:        {fail_hw}\n"
            f"FK Computation Failures:    {fail_fk}\n"
            f"============================================================\n"
        )
        
        with open(txt_file, "w") as f:
            f.write(summary_text)

        print(summary_text)
        print(f"💾 Saved Raw Execution Data to: {csv_file}")
        print(f"📄 Saved Results Summary to:    {txt_file}")
        
        # ✅ FIX 3: Check if success_df is empty before trying to plot
        if not success_df.empty:
            self.generate_publication_plot(success_df, timestamp)
        else:
            print("⚠️ 0 Successful Executions. Plot generation skipped.")

    def generate_publication_plot(self, df, timestamp):
        try:
            print("🎨 Generating auto-plot for paper...")
            plt.figure(figsize=(10, 6))
            
            types = df['test_type'].unique()
            data_to_plot = [df[df['test_type'] == t]['cartesian_error_mm'].dropna() for t in types]
            
            box = plt.boxplot(data_to_plot, labels=types, patch_artist=True)
            
            colors =['#4C72B0', '#DD8452'] 
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                
            plt.title('Hardware Validation: Positioning Error by Region', fontsize=14, fontweight='bold')
            plt.ylabel('Cartesian Error (mm)', fontsize=12, fontweight='bold')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plot_file = f"hardware_error_plot_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📈 Saved Paper-Ready Plot:      {plot_file}\n")
        except Exception as e:
            print(f"⚠️ Could not generate plot: {e}")

def main(args=None):
    rclpy.init(args=args)
    input_csv = "real_robot_40_test_poses_table_top.csv" 
    executor = RealRobotHardwareValidator(input_csv)
    executor.run_experiment()
    executor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()









# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient
# from control_msgs.action import FollowJointTrajectory
# from trajectory_msgs.msg import JointTrajectoryPoint
# from sensor_msgs.msg import JointState
# from moveit_msgs.srv import GetPositionFK, GetStateValidity  # ✅ ADDED GetStateValidity
# from moveit_msgs.msg import RobotState
# import pandas as pd
# import numpy as np
# import time
# from datetime import datetime
# import matplotlib.pyplot as plt

# class RealRobotHardwareValidator(Node):
#     def __init__(self, csv_file):
#         super().__init__('real_robot_hardware_validator')
        
#         print("\n" + "="*80)
#         print(" 🏆 PUBLICATION-GRADE HARDWARE VALIDATION FRAMEWORK")
#         print("="*80)
        
#         self.declare_parameter('controller_name', '/arm_controller/follow_joint_trajectory')
#         self.action_topic = self.get_parameter('controller_name').get_parameter_value().string_value
        
#         self.csv_file = csv_file
#         self.NUM_REPEATS = 3  
        
#         try:
#             self.df = pd.read_csv(self.csv_file)
#             print(f"[INFO] Loaded {len(self.df)} strategic test poses.")
#             print(f"[INFO] Repeatability Mode: {self.NUM_REPEATS} iterations per pose.")
#         except FileNotFoundError:
#             self.get_logger().error(f"Cannot find {self.csv_file}.")
#             exit()

#         self.trajectory_client = ActionClient(self, FollowJointTrajectory, self.action_topic)
#         self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
#         # ✅ FIX 2: Added State Validity Client for pre-execution checks
#         self.state_validity_client = self.create_client(GetStateValidity, "/check_state_validity")
        
#         self.actual_joints =[0.0, 0.0, 0.0, 0.0, 0.0]
#         self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

#         self.joint_names =['joint1', 'joint2', 'joint3', 'joint4', 'joint5_roll']
#         self.end_link = "end_effector_link"
#         self.HOME_POSE =[0.0, -1.0, 0.3, 0.7, 0.0]
        
#         # ✅ FIX 4: Safely lowered max jump from 3.5 (200deg) to 1.5 rad
#         self.MAX_SAFE_JUMP_RAD = 1.5  
        
#         self.joint_limits =[
#             (-3.14, 3.14),  # joint1
#             (-1.5, 1.5),    # joint2
#             (-1.5, 1.4),    # joint3
#             (-1.7, 1.97),   # joint4
#             (-3.0, 3.0)     # joint5
#         ]
        
#         self.execution_results =[]

#     def joint_state_callback(self, msg):
#         try:
#             for i, name in enumerate(self.joint_names):
#                 if name in msg.name:
#                     idx = msg.name.index(name)
#                     self.actual_joints[i] = msg.position[idx]
#         except ValueError:
#             pass

#     def wait_for_systems(self):
#         print(f"[INFO] Waiting for Action Server ({self.action_topic})...")
#         self.trajectory_client.wait_for_server()
#         print(f"[INFO] Waiting for FK Service (/compute_fk)...")
#         self.fk_client.wait_for_service()
#         # ✅ Added wait for collision check server
#         print(f"[INFO] Waiting for State Validity Service (/check_state_validity)...")
#         self.state_validity_client.wait_for_service()
#         print("[INFO] All Systems GO.\n")

#     def passes_safety_check(self, row):
#         """ 🔴 FAILURE CLASSIFICATION & SAFETY FILTER """
#         joints = np.array([row['j1'], row['j2'], row['j3'], row['j4'], row['j5']])
        
#         # 1. Joint Limits Check
#         for i, j_val in enumerate(joints):
#             min_lim, max_lim = self.joint_limits[i]
#             if not (min_lim <= j_val <= max_lim):
#                 return False, 'safety_block_limits'
                
#         # 2. Extreme Orientation Check 
#         # ✅ FIX 7: Relaxed constraint to only block pitch (table collision relevant)
#         if abs(row['pitch']) > 2.8:
#             return False, 'safety_block_orientation'
            
#         # 3. Extreme Jump Detection
#         max_jump = np.max(np.abs(joints - np.array(self.HOME_POSE)))
#         if max_jump > self.MAX_SAFE_JUMP_RAD:
#             return False, 'safety_block_jump'
            
#         # 4. Runtime Collision Validation Check (✅ FIX 2)
#         sv_req = GetStateValidity.Request()
#         sv_req.group_name = "arm"
#         rs = RobotState()
#         js = JointState(name=self.joint_names, position=joints.tolist())
#         rs.joint_state = js
#         sv_req.robot_state = rs
        
#         sv_future = self.state_validity_client.call_async(sv_req)
#         rclpy.spin_until_future_complete(self, sv_future)
#         sv_resp = sv_future.result()

#         if not sv_resp or not sv_resp.valid:
#             return False, 'safety_block_collision'
            
#         return True, 'safe'

#     def compute_actual_pose(self, joints_array):
#         req = GetPositionFK.Request()
#         req.header.frame_id = "link1" 
#         req.fk_link_names = [self.end_link]
        
#         rs = RobotState()
#         js = JointState(name=self.joint_names, position=joints_array)
#         rs.joint_state = js
#         req.robot_state = rs
        
#         future = self.fk_client.call_async(req)
#         rclpy.spin_until_future_complete(self, future)
#         resp = future.result()
        
#         if resp and resp.error_code.val == resp.error_code.SUCCESS:
#             p = resp.pose_stamped[0].pose
#             pos = np.array([p.position.x, p.position.y, p.position.z])
#             quat = np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
#             return pos, quat
#         return None, None

#     def execute_trajectory(self, target_joints, duration_sec=4.5):
#         goal_msg = FollowJointTrajectory.Goal()
#         goal_msg.trajectory.joint_names = self.joint_names
        
#         point = JointTrajectoryPoint()
#         point.positions = target_joints
        
#         # ✅ FIX 1: Safely constrained velocities and accelerations to prevent jerking
#         point.velocities = [0.5] * 5
#         point.accelerations = [0.5] * 5
        
#         point.time_from_start.sec = int(duration_sec)
#         point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
#         goal_msg.trajectory.points = [point]

#         send_goal_future = self.trajectory_client.send_goal_async(goal_msg)
#         rclpy.spin_until_future_complete(self, send_goal_future)
#         goal_handle = send_goal_future.result()

#         if not goal_handle.accepted:
#             return False

#         get_result_future = goal_handle.get_result_async()
#         rclpy.spin_until_future_complete(self, get_result_future)
#         result = get_result_future.result()
        
#         return result.result.error_code == 0

#     def run_experiment(self):
#         self.wait_for_systems()
#         total_poses = len(self.df)
        
#         # ✅ FIX 3: Keyboard Interrupt / Emergency Stop Handler Added
#         try:
#             for index, row in self.df.iterrows():
#                 print(f"\n[{index+1}/{total_poses}] Testing: {row['test_type']}")
                
#                 is_safe, failure_reason = self.passes_safety_check(row)
#                 if not is_safe:
#                     print(f"   ⚠️ Skipping entire pose. Reason: {failure_reason}")
#                     for i in range(self.NUM_REPEATS):
#                         self.log_data(index, row, i+1, 'failed', failure_reason)
#                     continue

#                 target_joints =[row['j1'], row['j2'], row['j3'], row['j4'], row['j5']]
                
#                 for iteration in range(self.NUM_REPEATS):
#                     self.execute_trajectory(self.HOME_POSE, duration_sec=3.0)
#                     time.sleep(0.5)

#                     success = self.execute_trajectory(target_joints, duration_sec=4.5)
                    
#                     if not success:
#                         print(f"   ❌ Hardware rejected trajectory on Iter {iteration+1}.")
#                         self.log_data(index, row, iteration+1, 'failed', 'hardware_rejected')
#                         continue
                    
#                     # ✅ FIX 6: Settle and read Actuals (Increased sleep and spin for accurate physical read)
#                     time.sleep(2.0)
#                     for _ in range(50):
#                         rclpy.spin_once(self, timeout_sec=0.05)
                        
#                     actual_j = self.actual_joints.copy()
#                     actual_xyz, actual_q = self.compute_actual_pose(actual_j)
                    
#                     if actual_xyz is None:
#                         print(f"   ❌ FK failed to compute actual pose on Iter {iteration+1}.")
#                         self.log_data(index, row, iteration+1, 'failed', 'fk_failed')
#                         continue
                    
#                     target_xyz = np.array([row['x'], row['y'], row['z']])
#                     target_q = np.array([row['qx'], row['qy'], row['qz'], row['qw']])

#                     cartesian_error_mm = np.linalg.norm(target_xyz - actual_xyz) * 1000.0
#                     mean_j_error = np.mean(np.abs(np.array(target_joints) - np.array(actual_j)))
                    
#                     dot = np.clip(np.abs(np.dot(target_q, actual_q)), 0.0, 1.0)
#                     angle_error_deg = np.degrees(2 * np.arccos(dot))
                    
#                     # ✅ FIX 6: Format to .6f to reveal sub-millimeter realistic physical errors
#                     print(f"   🔄 Iter {iteration+1} | Pos Err: {cartesian_error_mm:8.6f} mm | Ang Err: {angle_error_deg:8.6f}°")
                    
#                     self.log_data(index, row, iteration+1, 'success', 'none', actual_j, actual_xyz, actual_q, 
#                                   cartesian_error_mm, angle_error_deg, mean_j_error)

#             print("\n[INFO] Experiment Complete. Returning to HOME...")
#             self.execute_trajectory(self.HOME_POSE, duration_sec=3.0)

#         except KeyboardInterrupt:
#             print("\n🚨 EMERGENCY STOP TRIGGERED! Returning to HOME safely...")
#             self.execute_trajectory(self.HOME_POSE, duration_sec=3.0)
        
#         finally:
#             self.save_results()

#     def log_data(self, index, row, iteration, status, failure_type, 
#                  actual_j=None, actual_xyz=None, actual_q=None, 
#                  cartesian_err=np.nan, angular_err=np.nan, joint_err=np.nan):
        
#         self.execution_results.append({
#             'pose_id': index,
#             'iteration': iteration,
#             'test_type': row['test_type'],
#             'status': status,
#             'failure_type': failure_type,
            
#             'target_x': row['x'], 'target_y': row['y'], 'target_z': row['z'],
#             'target_qx': row['qx'], 'target_qy': row['qy'], 'target_qz': row['qz'], 'target_qw': row['qw'],
#             'cmd_j1': row['j1'], 'cmd_j2': row['j2'], 'cmd_j3': row['j3'], 'cmd_j4': row['j4'], 'cmd_j5': row['j5'],
            
#             'actual_x': actual_xyz[0] if actual_xyz is not None else np.nan,
#             'actual_y': actual_xyz[1] if actual_xyz is not None else np.nan,
#             'actual_z': actual_xyz[2] if actual_xyz is not None else np.nan,
#             'actual_qx': actual_q[0] if actual_q is not None else np.nan,
#             'actual_qy': actual_q[1] if actual_q is not None else np.nan,
#             'actual_qz': actual_q[2] if actual_q is not None else np.nan,
#             'actual_qw': actual_q[3] if actual_q is not None else np.nan,
#             'actual_j1': actual_j[0] if actual_j is not None else np.nan,
#             'actual_j2': actual_j[1] if actual_j is not None else np.nan,
#             'actual_j3': actual_j[2] if actual_j is not None else np.nan,
#             'actual_j4': actual_j[3] if actual_j is not None else np.nan,
#             'actual_j5': actual_j[4] if actual_j is not None else np.nan,

#             'cartesian_error_mm': cartesian_err,
#             'orientation_error_deg': angular_err,
#             'mean_joint_error_rad': joint_err
#         })

#     def save_results(self):
#         if len(self.execution_results) == 0:
#             print("❌ No successful executions to save.")
#             return
            
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         csv_file = f"hardware_validation_raw_data_{timestamp}.csv"
#         txt_file = f"hardware_validation_metrics_{timestamp}.txt"
        
#         results_df = pd.DataFrame(self.execution_results)
#         results_df.to_csv(csv_file, index=False)
        
#         success_df = results_df[results_df['status'] == 'success']
#         failed_df = results_df[results_df['status'] == 'failed']
        
#         total_attempts = len(results_df)
#         success_rate = (len(success_df) / total_attempts) * 100 if total_attempts > 0 else 0
        
#         avg_cartesian = success_df['cartesian_error_mm'].mean()
#         max_cartesian = success_df['cartesian_error_mm'].max()
#         min_cartesian = success_df['cartesian_error_mm'].min()
        
#         avg_angular = success_df['orientation_error_deg'].mean()
#         max_angular = success_df['orientation_error_deg'].max()
        
#         grouped = success_df.groupby('pose_id')['cartesian_error_mm']
#         std_dev_mean = grouped.std().mean()
#         std_dev_max = grouped.std().max()
        
#         fail_limits = len(failed_df[failed_df['failure_type'] == 'safety_block_limits'])
#         fail_orient = len(failed_df[failed_df['failure_type'] == 'safety_block_orientation'])
#         fail_jump = len(failed_df[failed_df['failure_type'] == 'safety_block_jump'])
#         # ✅ Tracked newly added explicit collision failures
#         fail_collision = len(failed_df[failed_df['failure_type'] == 'safety_block_collision'])
#         fail_hw = len(failed_df[failed_df['failure_type'] == 'hardware_rejected'])
#         fail_fk = len(failed_df[failed_df['failure_type'] == 'fk_failed'])

#         summary_text = (
#             f"============================================================\n"
#             f" 🎯 REAL-ROBOT HARDWARE VALIDATION & KINEMATIC METRICS\n"
#             f"============================================================\n"
#             f"Total Test Poses:           {len(self.df)}\n"
#             f"Iterations per Pose:        {self.NUM_REPEATS}\n"
#             f"Total Execution Attempts:   {total_attempts}\n"
#             f"Successful Executions:      {len(success_df)}\n"
#             f"Hardware Success Rate:      {success_rate:.2f}%\n"
#             f"------------------------------------------------------------\n"
#             f"📏 CARTESIAN POSITIONING (Accuracy)\n"
#             f"------------------------------------------------------------\n"
#             f"Mean Euclidean Error:       {avg_cartesian:7.4f} mm\n"
#             f"Max Euclidean Error:        {max_cartesian:7.4f} mm\n"
#             f"Min Euclidean Error:        {min_cartesian:7.4f} mm\n"
#             f"------------------------------------------------------------\n"
#             f"🔄 REPEATABILITY (Precision)\n"
#             f"------------------------------------------------------------\n"
#             f"Mean Spatial Std Dev:       ±{std_dev_mean:6.4f} mm\n"
#             f"Max Spatial Std Dev:        ±{std_dev_max:6.4f} mm\n"
#             f"------------------------------------------------------------\n"
#             f"📐 ORIENTATION FIDELITY\n"
#             f"------------------------------------------------------------\n"
#             f"Mean Angular Error:         {avg_angular:7.4f}°\n"
#             f"Max Angular Error:          {max_angular:7.4f}°\n"
#             f"------------------------------------------------------------\n"
#             f"⚠️ FAILURE ANALYSIS\n"
#             f"------------------------------------------------------------\n"
#             f"Total Failures:             {len(failed_df)}\n"
#             f"Safety Blocks (Limits):     {fail_limits}\n"
#             f"Safety Blocks (Orient/Jump):{fail_orient + fail_jump}\n"
#             f"Safety Blocks (Collision):  {fail_collision}\n"
#             f"Hardware Rejections:        {fail_hw}\n"
#             f"FK Computation Failures:    {fail_fk}\n"
#             f"============================================================\n"
#         )
        
#         with open(txt_file, "w") as f:
#             f.write(summary_text)

#         print(summary_text)
#         print(f"💾 Saved Raw Execution Data to: {csv_file}")
#         print(f"📄 Saved Results Summary to:    {txt_file}")
        
#         self.generate_publication_plot(success_df, timestamp)

#     def generate_publication_plot(self, df, timestamp):
#         try:
#             print("🎨 Generating auto-plot for paper...")
#             plt.figure(figsize=(10, 6))
            
#             types = df['test_type'].unique()
#             data_to_plot = [df[df['test_type'] == t]['cartesian_error_mm'].dropna() for t in types]
            
#             box = plt.boxplot(data_to_plot, labels=types, patch_artist=True)
            
#             colors =['#4C72B0', '#DD8452'] 
#             for patch, color in zip(box['boxes'], colors):
#                 patch.set_facecolor(color)
#                 patch.set_alpha(0.7)
                
#             plt.title('Hardware Validation: Positioning Error by Region', fontsize=14, fontweight='bold')
#             plt.ylabel('Cartesian Error (mm)', fontsize=12, fontweight='bold')
#             plt.grid(axis='y', linestyle='--', alpha=0.7)
            
#             plot_file = f"hardware_error_plot_{timestamp}.png"
#             plt.savefig(plot_file, dpi=300, bbox_inches='tight')
#             print(f"📈 Saved Paper-Ready Plot:      {plot_file}\n")
#         except Exception as e:
#             print(f"⚠️ Could not generate plot: {e}")

# def main(args=None):
#     rclpy.init(args=args)
#     input_csv = "real_robot_40_test_poses_table_top.csv" 
#     executor = RealRobotHardwareValidator(input_csv)
#     executor.run_experiment()
#     executor.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()








# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient
# from control_msgs.action import FollowJointTrajectory
# from trajectory_msgs.msg import JointTrajectoryPoint
# from sensor_msgs.msg import JointState
# from moveit_msgs.srv import GetPositionFK
# from moveit_msgs.msg import RobotState
# import pandas as pd
# import numpy as np
# import time
# from datetime import datetime
# import matplotlib.pyplot as plt

# class RealRobotHardwareValidator(Node):
#     def __init__(self, csv_file):
#         super().__init__('real_robot_hardware_validator')
        
#         print("\n" + "="*80)
#         print(" 🏆 PUBLICATION-GRADE HARDWARE VALIDATION FRAMEWORK")
#         print("="*80)
        
#         # --- 1. CONFIGURATION ---
#         self.declare_parameter('controller_name', '/arm_controller/follow_joint_trajectory')
#         self.action_topic = self.get_parameter('controller_name').get_parameter_value().string_value
        
#         self.csv_file = csv_file
#         self.NUM_REPEATS = 3  # ✅ Repeatability Metric (Variance Analysis)
        
#         try:
#             self.df = pd.read_csv(self.csv_file)
#             print(f"[INFO] Loaded {len(self.df)} strategic test poses.")
#             print(f"[INFO] Repeatability Mode: {self.NUM_REPEATS} iterations per pose.")
#         except FileNotFoundError:
#             self.get_logger().error(f"Cannot find {self.csv_file}.")
#             exit()

#         # --- 2. ROS 2 INTERFACES ---
#         self.trajectory_client = ActionClient(self, FollowJointTrajectory, self.action_topic)
#         self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        
#         self.actual_joints =[0.0, 0.0, 0.0, 0.0, 0.0]
#         self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

#         # --- 3. HARDWARE LIMITS & SAFETY ---
#         self.joint_names =['joint1', 'joint2', 'joint3', 'joint4', 'joint5_roll']
#         self.end_link = "end_effector_link"
#         self.HOME_POSE =[0.0, -1.0, 0.3, 0.7, 0.0]
#         self.MAX_SAFE_JUMP_RAD = 3.5  # Max allowed single-joint delta from Home
        
#         self.joint_limits =[
#             (-3.14, 3.14),  # joint1
#             (-1.5, 1.5),    # joint2
#             (-1.5, 1.4),    # joint3
#             (-1.7, 1.97),   # joint4
#             (-3.0, 3.0)     # joint5
#         ]
        
#         self.execution_results =[]

#     def joint_state_callback(self, msg):
#         try:
#             for i, name in enumerate(self.joint_names):
#                 if name in msg.name:
#                     idx = msg.name.index(name)
#                     self.actual_joints[i] = msg.position[idx]
#         except ValueError:
#             pass

#     def wait_for_systems(self):
#         print(f"[INFO] Waiting for Action Server ({self.action_topic})...")
#         self.trajectory_client.wait_for_server()
#         print(f"[INFO] Waiting for FK Service (/compute_fk)...")
#         self.fk_client.wait_for_service()
#         print("[INFO] All Systems GO.\n")

#     def passes_safety_check(self, row):
#         """ 🔴 FAILURE CLASSIFICATION & SAFETY FILTER """
#         joints = np.array([row['j1'], row['j2'], row['j3'], row['j4'], row['j5']])
        
#         # 1. Joint Limits Check
#         for i, j_val in enumerate(joints):
#             min_lim, max_lim = self.joint_limits[i]
#             if not (min_lim <= j_val <= max_lim):
#                 return False, 'safety_block_limits'
                
#         # 2. Extreme Orientation Check
#         if abs(row['roll']) > 2.8 or abs(row['pitch']) > 2.8 or abs(row['yaw']) > 2.8:
#             return False, 'safety_block_orientation'
            
#         # 3. Extreme Jump Detection
#         max_jump = np.max(np.abs(joints - np.array(self.HOME_POSE)))
#         if max_jump > self.MAX_SAFE_JUMP_RAD:
#             return False, 'safety_block_jump'
            
#         return True, 'safe'

#     def compute_actual_pose(self, joints_array):
#         """ Computes Actual XYZ and Actual Quaternion natively from Joint states """
#         req = GetPositionFK.Request()
#         req.header.frame_id = "link1" 
#         req.fk_link_names = [self.end_link]
        
#         rs = RobotState()
#         js = JointState(name=self.joint_names, position=joints_array)
#         rs.joint_state = js
#         req.robot_state = rs
        
#         future = self.fk_client.call_async(req)
#         rclpy.spin_until_future_complete(self, future)
#         resp = future.result()
        
#         if resp and resp.error_code.val == resp.error_code.SUCCESS:
#             p = resp.pose_stamped[0].pose
#             pos = np.array([p.position.x, p.position.y, p.position.z])
#             quat = np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
#             return pos, quat
#         return None, None

#     def execute_trajectory(self, target_joints, duration_sec=4.5):
#         goal_msg = FollowJointTrajectory.Goal()
#         goal_msg.trajectory.joint_names = self.joint_names
        
#         point = JointTrajectoryPoint()
#         point.positions = target_joints
#         point.velocities = [0.0] * 5
#         point.accelerations = [0.0] * 5
        
#         point.time_from_start.sec = int(duration_sec)
#         point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
#         goal_msg.trajectory.points = [point]

#         send_goal_future = self.trajectory_client.send_goal_async(goal_msg)
#         rclpy.spin_until_future_complete(self, send_goal_future)
#         goal_handle = send_goal_future.result()

#         if not goal_handle.accepted:
#             return False

#         get_result_future = goal_handle.get_result_async()
#         rclpy.spin_until_future_complete(self, get_result_future)
#         result = get_result_future.result()
        
#         return result.result.error_code == 0

#     def run_experiment(self):
#         self.wait_for_systems()
#         total_poses = len(self.df)
        
#         for index, row in self.df.iterrows():
#             print(f"\n[{index+1}/{total_poses}] Testing: {row['test_type']}")
            
#             # --- 1. SAFETY FILTER ---
#             is_safe, failure_reason = self.passes_safety_check(row)
#             if not is_safe:
#                 print(f"   ⚠️ Skipping entire pose. Reason: {failure_reason}")
#                 # Log failure for all planned iterations
#                 for i in range(self.NUM_REPEATS):
#                     self.log_data(index, row, i+1, 'failed', failure_reason)
#                 continue

#             target_joints =[row['j1'], row['j2'], row['j3'], row['j4'], row['j5']]
            
#             # --- 2. EXECUTION LOOP ---
#             for iteration in range(self.NUM_REPEATS):
#                 self.execute_trajectory(self.HOME_POSE, duration_sec=3.0)
#                 time.sleep(0.5)

#                 success = self.execute_trajectory(target_joints, duration_sec=4.5)
                
#                 if not success:
#                     print(f"   ❌ Hardware rejected trajectory on Iter {iteration+1}.")
#                     self.log_data(index, row, iteration+1, 'failed', 'hardware_rejected')
#                     continue
                
#                 # Settle and read Actuals
#                 time.sleep(1.0)
#                 for _ in range(10):
#                     rclpy.spin_once(self, timeout_sec=0.05)
                    
#                 actual_j = self.actual_joints.copy()
#                 actual_xyz, actual_q = self.compute_actual_pose(actual_j)
                
#                 if actual_xyz is None:
#                     print(f"   ❌ FK failed to compute actual pose on Iter {iteration+1}.")
#                     self.log_data(index, row, iteration+1, 'failed', 'fk_failed')
#                     continue
                
#                 # --- 3. METRICS COMPUTATION ---
#                 target_xyz = np.array([row['x'], row['y'], row['z']])
#                 target_q = np.array([row['qx'], row['qy'], row['qz'], row['qw']])

#                 cartesian_error_mm = np.linalg.norm(target_xyz - actual_xyz) * 1000.0
#                 mean_j_error = np.mean(np.abs(np.array(target_joints) - np.array(actual_j)))
                
#                 # Quaternion Dot Product Orientation Error
#                 dot = np.clip(np.abs(np.dot(target_q, actual_q)), 0.0, 1.0)
#                 angle_error_deg = np.degrees(2 * np.arccos(dot))
                
#                 print(f"   🔄 Iter {iteration+1} | Pos Err: {cartesian_error_mm:5.2f} mm | Ang Err: {angle_error_deg:5.2f}°")
                
#                 # --- 4. SUCCESSFUL LOGGING ---
#                 self.log_data(index, row, iteration+1, 'success', 'none', actual_j, actual_xyz, actual_q, 
#                               cartesian_error_mm, angle_error_deg, mean_j_error)

#         print("\n[INFO] Experiment Complete. Returning to HOME...")
#         self.execute_trajectory(self.HOME_POSE, duration_sec=3.0)
#         self.save_results()

#     def log_data(self, index, row, iteration, status, failure_type, 
#                  actual_j=None, actual_xyz=None, actual_q=None, 
#                  cartesian_err=np.nan, angular_err=np.nan, joint_err=np.nan):
        
#         # ✅ NEW: Safe explicit tracking of Raw Data for Post-Processing
#         self.execution_results.append({
#             'pose_id': index,
#             'iteration': iteration,
#             'test_type': row['test_type'],
#             'status': status,
#             'failure_type': failure_type,
            
#             # Targets
#             'target_x': row['x'], 'target_y': row['y'], 'target_z': row['z'],
#             'target_qx': row['qx'], 'target_qy': row['qy'], 'target_qz': row['qz'], 'target_qw': row['qw'],
#             'cmd_j1': row['j1'], 'cmd_j2': row['j2'], 'cmd_j3': row['j3'], 'cmd_j4': row['j4'], 'cmd_j5': row['j5'],
            
#             # Actuals (Safely handled if None)
#             'actual_x': actual_xyz[0] if actual_xyz is not None else np.nan,
#             'actual_y': actual_xyz[1] if actual_xyz is not None else np.nan,
#             'actual_z': actual_xyz[2] if actual_xyz is not None else np.nan,
#             'actual_qx': actual_q[0] if actual_q is not None else np.nan,
#             'actual_qy': actual_q[1] if actual_q is not None else np.nan,
#             'actual_qz': actual_q[2] if actual_q is not None else np.nan,
#             'actual_qw': actual_q[3] if actual_q is not None else np.nan,
#             'actual_j1': actual_j[0] if actual_j is not None else np.nan,
#             'actual_j2': actual_j[1] if actual_j is not None else np.nan,
#             'actual_j3': actual_j[2] if actual_j is not None else np.nan,
#             'actual_j4': actual_j[3] if actual_j is not None else np.nan,
#             'actual_j5': actual_j[4] if actual_j is not None else np.nan,

#             # Errors
#             'cartesian_error_mm': cartesian_err,
#             'orientation_error_deg': angular_err,
#             'mean_joint_error_rad': joint_err
#         })

#     def save_results(self):
#         if len(self.execution_results) == 0:
#             print("❌ No successful executions to save.")
#             return
            
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         csv_file = f"hardware_validation_raw_data_{timestamp}.csv"
#         txt_file = f"hardware_validation_metrics_{timestamp}.txt"
        
#         results_df = pd.DataFrame(self.execution_results)
#         results_df.to_csv(csv_file, index=False)
        
#         # --- CALCULATE FINAL RESEARCH METRICS ---
#         success_df = results_df[results_df['status'] == 'success']
#         failed_df = results_df[results_df['status'] == 'failed']
        
#         total_attempts = len(results_df)
#         success_rate = (len(success_df) / total_attempts) * 100 if total_attempts > 0 else 0
        
#         # Cartesian Accuracy
#         avg_cartesian = success_df['cartesian_error_mm'].mean()
#         max_cartesian = success_df['cartesian_error_mm'].max()
#         min_cartesian = success_df['cartesian_error_mm'].min()
        
#         # Orientation Accuracy
#         avg_angular = success_df['orientation_error_deg'].mean()
#         max_angular = success_df['orientation_error_deg'].max()
        
#         # Repeatability (Standard Deviation per Pose)
#         grouped = success_df.groupby('pose_id')['cartesian_error_mm']
#         std_dev_mean = grouped.std().mean()
#         std_dev_max = grouped.std().max()
        
#         # Failure Breakdown
#         fail_limits = len(failed_df[failed_df['failure_type'] == 'safety_block_limits'])
#         fail_orient = len(failed_df[failed_df['failure_type'] == 'safety_block_orientation'])
#         fail_jump = len(failed_df[failed_df['failure_type'] == 'safety_block_jump'])
#         fail_hw = len(failed_df[failed_df['failure_type'] == 'hardware_rejected'])
#         fail_fk = len(failed_df[failed_df['failure_type'] == 'fk_failed'])

#         # --- GENERATE PUBLICATION-READY TEXT SUMMARY ---
#         summary_text = (
#             f"============================================================\n"
#             f" 🎯 REAL-ROBOT HARDWARE VALIDATION & KINEMATIC METRICS\n"
#             f"============================================================\n"
#             f"Total Test Poses:           {len(self.df)}\n"
#             f"Iterations per Pose:        {self.NUM_REPEATS}\n"
#             f"Total Execution Attempts:   {total_attempts}\n"
#             f"Successful Executions:      {len(success_df)}\n"
#             f"Hardware Success Rate:      {success_rate:.2f}%\n"
#             f"------------------------------------------------------------\n"
#             f"📏 CARTESIAN POSITIONING (Accuracy)\n"
#             f"------------------------------------------------------------\n"
#             f"Mean Euclidean Error:       {avg_cartesian:7.2f} mm\n"
#             f"Max Euclidean Error:        {max_cartesian:7.2f} mm\n"
#             f"Min Euclidean Error:        {min_cartesian:7.2f} mm\n"
#             f"------------------------------------------------------------\n"
#             f"🔄 REPEATABILITY (Precision)\n"
#             f"------------------------------------------------------------\n"
#             f"Mean Spatial Std Dev:       ±{std_dev_mean:6.2f} mm\n"
#             f"Max Spatial Std Dev:        ±{std_dev_max:6.2f} mm\n"
#             f"------------------------------------------------------------\n"
#             f"📐 ORIENTATION FIDELITY\n"
#             f"------------------------------------------------------------\n"
#             f"Mean Angular Error:         {avg_angular:7.2f}°\n"
#             f"Max Angular Error:          {max_angular:7.2f}°\n"
#             f"------------------------------------------------------------\n"
#             f"⚠️ FAILURE ANALYSIS\n"
#             f"------------------------------------------------------------\n"
#             f"Total Failures:             {len(failed_df)}\n"
#             f"Safety Blocks (Limits):     {fail_limits}\n"
#             f"Safety Blocks (Orient/Jump):{fail_orient + fail_jump}\n"
#             f"Hardware Rejections:        {fail_hw}\n"
#             f"FK Computation Failures:    {fail_fk}\n"
#             f"============================================================\n"
#         )
        
#         # Save Text File
#         with open(txt_file, "w") as f:
#             f.write(summary_text)

#         # Print to Console
#         print(summary_text)
#         print(f"💾 Saved Raw Execution Data to: {csv_file}")
#         print(f"📄 Saved Results Summary to:    {txt_file}")
        
#         self.generate_publication_plot(success_df, timestamp)

#     def generate_publication_plot(self, df, timestamp):
#         try:
#             print("🎨 Generating auto-plot for paper...")
#             plt.figure(figsize=(10, 6))
            
#             types = df['test_type'].unique()
#             data_to_plot = [df[df['test_type'] == t]['cartesian_error_mm'].dropna() for t in types]
            
#             box = plt.boxplot(data_to_plot, labels=types, patch_artist=True)
            
#             colors =['#4C72B0', '#DD8452'] # Research/Seaborn classic colors
#             for patch, color in zip(box['boxes'], colors):
#                 patch.set_facecolor(color)
#                 patch.set_alpha(0.7)
                
#             plt.title('Hardware Validation: Positioning Error by Region', fontsize=14, fontweight='bold')
#             plt.ylabel('Cartesian Error (mm)', fontsize=12, fontweight='bold')
#             plt.grid(axis='y', linestyle='--', alpha=0.7)
            
#             plot_file = f"hardware_error_plot_{timestamp}.png"
#             plt.savefig(plot_file, dpi=300, bbox_inches='tight')
#             print(f"📈 Saved Paper-Ready Plot:      {plot_file}\n")
#         except Exception as e:
#             print(f"⚠️ Could not generate plot: {e}")

# def main(args=None):
#     rclpy.init(args=args)
#     input_csv = "real_robot_40_test_poses_above_table.csv" 
#     executor = RealRobotHardwareValidator(input_csv)
#     executor.run_experiment()
#     executor.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()