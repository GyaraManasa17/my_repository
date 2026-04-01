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
        
        self.SUCCESS_POS_THRESH = 5.0    
        self.SUCCESS_ORIENT_THRESH = 5.0 
        
        # ✅ FIX 1: PHASED TESTING TOGGLE
        # Set to False ONLY when you are ready to test boundary (extreme) poses.
        self.PHASE_1_SAFE_MODE = False
        
        try:
            self.df = pd.read_csv(self.csv_file)
            if self.PHASE_1_SAFE_MODE:
                print("⚠️ [PHASE 1 SAFE MODE ACTIVE] Filtering out boundary points!")
                self.df = self.df[~self.df['test_type'].str.contains('Boundary')]
            
            # ✅ FIX 2: Only test the first 5 poses if Phase 1 is active
            if self.PHASE_1_SAFE_MODE and len(self.df) > 5:
                self.df = self.df.head(5)
                print(f"⚠️ [PHASE 1 SAFE MODE ACTIVE] Reduced testing to FIRST 5 POSES only.")
                
            print(f"[INFO] Loaded {len(self.df)} strategic test poses to execute.")
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
        self.MAX_SAFE_JUMP_RAD = 2.5  
        
        self.joint_limits =[
            (-3.14, 3.14), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97), (-3.0, 3.0)     
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
        print(f"[INFO] Waiting for Services to come online...")
        self.trajectory_client.wait_for_server()
        self.fk_client.wait_for_service()
        self.state_validity_client.wait_for_service()
        print("[INFO] All Systems GO.\n")

    def passes_safety_check(self, row, hover_joints):
        if row['z'] < 0.20:
            return False, 'safety_block_low_z'
            
        joints = np.array([row['j1'], row['j2'], row['j3'], row['j4'], row['j5']])
        
        for i, j_val in enumerate(joints):
            min_lim, max_lim = self.joint_limits[i]
            if not (min_lim <= j_val <= max_lim):
                return False, 'safety_block_limits'
                
        if abs(row['pitch']) > 2.8:
            return False, 'safety_block_orientation'
            
        # ✅ FIX 3: Clamp Wrist Wrap Check
        if abs(row['j5']) > 2.5:
            return False, 'safety_block_wrist_wrap'
            
        # ✅ FIX 4: Check Delta between HOVER and TARGET (True sequential execution path)
        hover_to_target_jump = np.max(np.abs(joints - np.array(hover_joints)))
        if hover_to_target_jump > self.MAX_SAFE_JUMP_RAD:
            return False, 'safety_block_sequential_jump'
            
        sv_req = GetStateValidity.Request()
        sv_req.group_name = "arm"
        rs = RobotState()
        js = JointState(name=self.joint_names, position=joints.tolist())
        rs.joint_state = js
        sv_req.robot_state = rs
        
        sv_future = self.state_validity_client.call_async(sv_req)
        rclpy.spin_until_future_complete(self, sv_future, timeout_sec=2.0)
        if not sv_future.done(): return False, 'sv_timeout'
        
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
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done(): return None, None
        
        resp = future.result()
        if resp and resp.error_code.val == resp.error_code.SUCCESS:
            p = resp.pose_stamped[0].pose
            pos = np.array([p.position.x, p.position.y, p.position.z])
            quat = np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
            return pos, quat
        return None, None

    def execute_trajectory(self, target_joints, duration_sec):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.velocities = [0.0] * 5
        point.accelerations = [0.0] * 5
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
        goal_msg.trajectory.points = [point]

        send_goal_future = self.trajectory_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted: return False
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result()
        return result.result.error_code == 0

    def run_experiment(self):
        self.wait_for_systems()
        total_poses = len(self.df)
        
        try:
            print("\n[INFO] Moving to HOME pose to begin...")
            # ✅ FIX 5: Slower home initialization
            self.execute_trajectory(self.HOME_POSE, duration_sec=5.0)
            time.sleep(1.0)

            for index, row in self.df.iterrows():
                print(f"\n[{index+1}/{total_poses}] Testing: {row['test_type']}")
                
                target_joints =[row['j1'], row['j2'], row['j3'], row['j4'], row['j5']]
                hover_joints = [target_joints[0], self.HOME_POSE[1], self.HOME_POSE[2], self.HOME_POSE[3], target_joints[4]]
                
                # Check safety WITH respect to the hover position
                is_safe, failure_reason = self.passes_safety_check(row, hover_joints)
                if not is_safe:
                    print(f"   ⚠️ Skipping entire pose. Reason: {failure_reason}")
                    for i in range(self.NUM_REPEATS):
                        self.log_data(index, row, i+1, 'failed', failure_reason)
                    continue

                for iteration in range(self.NUM_REPEATS):
                    # ✅ FIX 6: MASSIVELY SLOWER Execution Times for Torque management
                    hover_duration = 8.0 if iteration == 0 else 5.0
                    self.execute_trajectory(hover_joints, duration_sec=hover_duration)
                    time.sleep(1.5)

                    start_exec_time = time.time()
                    success = self.execute_trajectory(target_joints, duration_sec=6.0)
                    exec_time = time.time() - start_exec_time
                    
                    if not success:
                        print(f"   ❌ Hardware rejected trajectory on Iter {iteration+1}.")
                        self.log_data(index, row, iteration+1, 'failed', 'hardware_rejected')
                        continue
                    
                    prev_j = None
                    stable = False
                    for _ in range(30):
                        rclpy.spin_once(self, timeout_sec=0.05)
                        if prev_j is not None:
                            if np.linalg.norm(np.array(self.actual_joints) - prev_j) < 0.0005:
                                stable = True
                                break
                        prev_j = np.array(self.actual_joints)
                        
                    if not stable: print("   ⚠️ Robot not fully settled, measurement may be noisy.")
                        
                    actual_j = self.actual_joints.copy()
                    actual_xyz, actual_q = self.compute_actual_pose(actual_j)
                    
                    if actual_xyz is None:
                        self.log_data(index, row, iteration+1, 'failed', 'fk_failed')
                        continue
                    
                    target_xyz = np.array([row['x'], row['y'], row['z']])
                    target_q = np.array([row['qx'], row['qy'], row['qz'], row['qw']])

                    target_q = target_q / np.linalg.norm(target_q)
                    actual_q = actual_q / np.linalg.norm(actual_q)

                    cartesian_error_mm = np.linalg.norm(target_xyz - actual_xyz) * 1000.0
                    joint_tracking_error = np.linalg.norm(np.array(target_joints) - np.array(actual_j))
                    mean_j_error = np.mean(np.abs(np.array(target_joints) - np.array(actual_j)))
                    
                    dot = np.clip(np.abs(np.dot(target_q, actual_q)), 0.0, 1.0)
                    angle_error_deg = np.degrees(2 * np.arccos(dot))
                    
                    if cartesian_error_mm > self.SUCCESS_POS_THRESH or angle_error_deg > self.SUCCESS_ORIENT_THRESH:
                        status = 'inaccurate'
                    else:
                        status = 'success'
                    
                    print(f"   🔄 Iter {iteration+1} [{status.upper()}] | Pos Err: {cartesian_error_mm:6.3f} mm | Ang Err: {angle_error_deg:6.3f}° | Time: {exec_time:.2f}s")
                    
                    self.log_data(index, row, iteration+1, status, 'none', actual_j, actual_xyz, actual_q, 
                                  cartesian_error_mm, angle_error_deg, mean_j_error, exec_time, joint_tracking_error)
                    
                    self.execute_trajectory(hover_joints, duration_sec=5.0)

            print("\n[INFO] Experiment Complete. Returning to HOME...")
            self.execute_trajectory(self.HOME_POSE, duration_sec=5.0)

        except KeyboardInterrupt:
            print("\n🚨 EMERGENCY STOP TRIGGERED! Returning to HOME safely...")
            safe_up = [self.actual_joints[0], -1.2, 0.0, 0.0, 0.0]
            self.execute_trajectory(safe_up, duration_sec=4.0)
            self.execute_trajectory(self.HOME_POSE, duration_sec=5.0)
        finally:
            self.save_results()

    def log_data(self, index, row, iteration, status, failure_type, 
                 actual_j=None, actual_xyz=None, actual_q=None, 
                 cartesian_err=np.nan, angular_err=np.nan, joint_err=np.nan,
                 exec_time=np.nan, tracking_err=np.nan):
        
        self.execution_results.append({
            'execution_order': index, 
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
            
            'cartesian_error_mm': cartesian_err,
            'orientation_error_deg': angular_err,
            'mean_joint_error_rad': joint_err,
            'joint_tracking_error_rad': tracking_err, 
            'execution_time_sec': exec_time           
        })

    def save_results(self):
        if len(self.execution_results) == 0:
            print("❌ No executions to save.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"hardware_validation_raw_data_{timestamp}.csv"
        txt_file = f"hardware_validation_metrics_{timestamp}.txt"
        
        results_df = pd.DataFrame(self.execution_results)
        results_df.to_csv(csv_file, index=False)
        
        success_df = results_df[results_df['status'] == 'success']
        inacc_df = results_df[results_df['status'] == 'inaccurate']
        valid_df = pd.concat([success_df, inacc_df]) 
        failed_df = results_df[results_df['status'] == 'failed']
        
        total_attempts = len(results_df)
        success_rate = (len(success_df) / total_attempts) * 100 if total_attempts > 0 else 0
        
        planning_failures = len(failed_df[failed_df['failure_type'].isin(['safety_block_collision', 'sv_timeout', 'fk_failed', 'safety_block_sequential_jump'])])
        ik_feasibility_rate = ((total_attempts - planning_failures) / total_attempts) * 100 if total_attempts > 0 else 0
        
        if not valid_df.empty:
            orientation_success_rate = (valid_df['orientation_error_deg'] < self.SUCCESS_ORIENT_THRESH).mean() * 100
        else:
            orientation_success_rate = 0.0
            
        bound_df = valid_df[valid_df['test_type'].str.contains('Boundary')]
        int_df = valid_df[valid_df['test_type'].str.contains('Interior')]
        
        bound_err = bound_df['cartesian_error_mm'].mean() if not bound_df.empty else np.nan
        int_err = int_df['cartesian_error_mm'].mean() if not int_df.empty else np.nan
        
        bound_succ = (len(success_df[success_df['test_type'].str.contains('Boundary')]) / len(results_df[results_df['test_type'].str.contains('Boundary')])) * 100 if len(results_df[results_df['test_type'].str.contains('Boundary')]) > 0 else np.nan
        int_succ = (len(success_df[success_df['test_type'].str.contains('Interior')]) / len(results_df[results_df['test_type'].str.contains('Interior')])) * 100 if len(results_df[results_df['test_type'].str.contains('Interior')]) > 0 else np.nan
        
        avg_cartesian = valid_df['cartesian_error_mm'].mean() if not valid_df.empty else np.nan
        max_cartesian = valid_df['cartesian_error_mm'].max() if not valid_df.empty else np.nan
        
        avg_angular = valid_df['orientation_error_deg'].mean() if not valid_df.empty else np.nan
        avg_time = valid_df['execution_time_sec'].mean() if not valid_df.empty else np.nan
        
        grouped = valid_df.groupby('pose_id')['cartesian_error_mm']
        std_dev_mean = grouped.std().mean() if not valid_df.empty else np.nan
        
        summary_text = (
            f"============================================================\n"
            f" 🎯 REAL-ROBOT HARDWARE VALIDATION METRICS (RESEARCH FORMAT)\n"
            f"============================================================\n"
            f"Total Target Executions:        {total_attempts}\n"
            f"Sim-to-Real Feasibility Rate:   {ik_feasibility_rate:.2f}% (Passed Planning)\n"
            f"Strict Execution Success Rate:  {success_rate:.2f}% (Error < 5mm)\n"
            f"Orientation Constraint Success: {orientation_success_rate:.2f}% (Error < 5°)\n"
            f"------------------------------------------------------------\n"
            f"📏 SIM-TO-REAL CARTESIAN GAP (Accuracy)\n"
            f"------------------------------------------------------------\n"
            f"Mean Euclidean Gap:             {avg_cartesian:7.4f} mm\n"
            f"Max Euclidean Gap:              {max_cartesian:7.4f} mm\n"
            f"------------------------------------------------------------\n"
            f"🗺 REGIONAL CAPABILITY (Boundary vs Interior)\n"
            f"------------------------------------------------------------\n"
            f"Boundary Mean Error:            {bound_err:7.4f} mm\n"
            f"Interior Mean Error:            {int_err:7.4f} mm\n"
            f"Boundary Success Rate:          {bound_succ:7.2f} %\n"
            f"Interior Success Rate:          {int_succ:7.2f} %\n"
            f"------------------------------------------------------------\n"
            f"🔄 HARDWARE REPEATABILITY (Precision)\n"
            f"------------------------------------------------------------\n"
            f"Mean Spatial Std Dev:           ±{std_dev_mean:6.4f} mm\n"
            f"------------------------------------------------------------\n"
            f"📐 6-DOF ORIENTATION FIDELITY\n"
            f"------------------------------------------------------------\n"
            f"Mean Angular Error:             {avg_angular:7.4f}°\n"
            f"------------------------------------------------------------\n"
            f"⏱ KINEMATIC PERFORMANCE\n"
            f"------------------------------------------------------------\n"
            f"Average Trajectory Exec Time:   {avg_time:7.3f} sec\n"
            f"============================================================\n"
        )
        
        with open(txt_file, "w") as f:
            f.write(summary_text)

        print(summary_text)
        print(f"💾 Saved Raw Execution Data to: {csv_file}")
        print(f"📄 Saved Results Summary to:    {txt_file}")
        
        if not valid_df.empty:
            self.generate_publication_plot(valid_df, timestamp)

    def generate_publication_plot(self, df, timestamp):
        try:
            plt.figure(figsize=(10, 6))
            types = df['test_type'].unique()
            data_to_plot = [df[df['test_type'] == t]['cartesian_error_mm'].dropna() for t in types]
            box = plt.boxplot(data_to_plot, labels=types, patch_artist=True)
            colors =['#4C72B0', '#DD8452'] 
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            plt.title('Sim-to-Real Validation: Positioning Error by Region', fontsize=14, fontweight='bold')
            plt.ylabel('Sim-to-Real Cartesian Gap (mm)', fontsize=12, fontweight='bold')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plot_file = f"hardware_error_plot_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📈 Saved Paper-Ready Plot:      {plot_file}\n")
        except Exception: pass

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
# from moveit_msgs.srv import GetPositionFK, GetStateValidity
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
        
#         self.SUCCESS_POS_THRESH = 5.0    # mm
#         self.SUCCESS_ORIENT_THRESH = 5.0 # deg
        
#         try:
#             self.df = pd.read_csv(self.csv_file)
#             print(f"[INFO] Loaded {len(self.df)} strategic test poses.")
#         except FileNotFoundError:
#             self.get_logger().error(f"Cannot find {self.csv_file}.")
#             exit()

#         self.trajectory_client = ActionClient(self, FollowJointTrajectory, self.action_topic)
#         self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
#         self.state_validity_client = self.create_client(GetStateValidity, "/check_state_validity")
        
#         self.actual_joints =[0.0, 0.0, 0.0, 0.0, 0.0]
#         self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

#         self.joint_names =['joint1', 'joint2', 'joint3', 'joint4', 'joint5_roll']
#         self.end_link = "end_effector_link"
#         self.HOME_POSE =[0.0, -1.0, 0.3, 0.7, 0.0]
#         self.MAX_SAFE_JUMP_RAD = 2.5  
        
#         self.joint_limits =[
#             (-3.14, 3.14), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97), (-3.0, 3.0)     
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
#         print(f"[INFO] Waiting for Services to come online...")
#         self.trajectory_client.wait_for_server()
#         self.fk_client.wait_for_service()
#         self.state_validity_client.wait_for_service()
#         print("[INFO] All Systems GO.\n")

#     def passes_safety_check(self, row):
#         if row['z'] < 0.18:
#             return False, 'safety_block_low_z'
            
#         joints = np.array([row['j1'], row['j2'], row['j3'], row['j4'], row['j5']])
        
#         for i, j_val in enumerate(joints):
#             min_lim, max_lim = self.joint_limits[i]
#             if not (min_lim <= j_val <= max_lim):
#                 return False, 'safety_block_limits'
                
#         if abs(row['pitch']) > 2.8:
#             return False, 'safety_block_orientation'
            
#         max_jump = np.max(np.abs(joints - np.array(self.HOME_POSE)))
#         if max_jump > self.MAX_SAFE_JUMP_RAD:
#             return False, 'safety_block_jump'
            
#         sv_req = GetStateValidity.Request()
#         sv_req.group_name = "arm"
#         rs = RobotState()
#         js = JointState(name=self.joint_names, position=joints.tolist())
#         rs.joint_state = js
#         sv_req.robot_state = rs
        
#         sv_future = self.state_validity_client.call_async(sv_req)
#         rclpy.spin_until_future_complete(self, sv_future, timeout_sec=2.0)
#         if not sv_future.done(): return False, 'sv_timeout'
        
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
#         rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
#         if not future.done(): return None, None
        
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

#         if not goal_handle.accepted: return False
#         get_result_future = goal_handle.get_result_async()
#         rclpy.spin_until_future_complete(self, get_result_future)
#         result = get_result_future.result()
#         return result.result.error_code == 0

#     def run_experiment(self):
#         self.wait_for_systems()
#         total_poses = len(self.df)
        
#         try:
#             print("\n[INFO] Moving to HOME pose to begin...")
#             self.execute_trajectory(self.HOME_POSE, duration_sec=3.0)
#             time.sleep(1.0)

#             for index, row in self.df.iterrows():
#                 print(f"\n[{index+1}/{total_poses}] Testing: {row['test_type']}")
                
#                 is_safe, failure_reason = self.passes_safety_check(row)
#                 if not is_safe:
#                     print(f"   ⚠️ Skipping entire pose. Reason: {failure_reason}")
#                     for i in range(self.NUM_REPEATS):
#                         self.log_data(index, row, i+1, 'failed', failure_reason)
#                     continue

#                 target_joints =[row['j1'], row['j2'], row['j3'], row['j4'], row['j5']]
#                 hover_joints = [target_joints[0], self.HOME_POSE[1], self.HOME_POSE[2], self.HOME_POSE[3], target_joints[4]]

#                 for iteration in range(self.NUM_REPEATS):
#                     hover_duration = 6.0 if iteration == 0 else 2.5
#                     self.execute_trajectory(hover_joints, duration_sec=hover_duration)
#                     time.sleep(0.2)

#                     start_exec_time = time.time()
#                     success = self.execute_trajectory(target_joints, duration_sec=3.0)
#                     exec_time = time.time() - start_exec_time
                    
#                     if not success:
#                         print(f"   ❌ Hardware rejected trajectory on Iter {iteration+1}.")
#                         self.log_data(index, row, iteration+1, 'failed', 'hardware_rejected')
#                         continue
                    
#                     prev_j = None
#                     stable = False
#                     for _ in range(30):
#                         rclpy.spin_once(self, timeout_sec=0.05)
#                         if prev_j is not None:
#                             if np.linalg.norm(np.array(self.actual_joints) - prev_j) < 0.0005:
#                                 stable = True
#                                 break
#                         prev_j = np.array(self.actual_joints)
                        
#                     if not stable: print("   ⚠️ Robot not fully settled, measurement may be noisy.")
                        
#                     actual_j = self.actual_joints.copy()
#                     actual_xyz, actual_q = self.compute_actual_pose(actual_j)
                    
#                     if actual_xyz is None:
#                         self.log_data(index, row, iteration+1, 'failed', 'fk_failed')
#                         continue
                    
#                     target_xyz = np.array([row['x'], row['y'], row['z']])
#                     target_q = np.array([row['qx'], row['qy'], row['qz'], row['qw']])

#                     target_q = target_q / np.linalg.norm(target_q)
#                     actual_q = actual_q / np.linalg.norm(actual_q)

#                     cartesian_error_mm = np.linalg.norm(target_xyz - actual_xyz) * 1000.0
#                     joint_tracking_error = np.linalg.norm(np.array(target_joints) - np.array(actual_j))
#                     mean_j_error = np.mean(np.abs(np.array(target_joints) - np.array(actual_j)))
                    
#                     dot = np.clip(np.abs(np.dot(target_q, actual_q)), 0.0, 1.0)
#                     angle_error_deg = np.degrees(2 * np.arccos(dot))
                    
#                     if cartesian_error_mm > self.SUCCESS_POS_THRESH or angle_error_deg > self.SUCCESS_ORIENT_THRESH:
#                         status = 'inaccurate'
#                     else:
#                         status = 'success'
                    
#                     print(f"   🔄 Iter {iteration+1} [{status.upper()}] | Pos Err: {cartesian_error_mm:6.3f} mm | Ang Err: {angle_error_deg:6.3f}° | Time: {exec_time:.2f}s")
                    
#                     self.log_data(index, row, iteration+1, status, 'none', actual_j, actual_xyz, actual_q, 
#                                   cartesian_error_mm, angle_error_deg, mean_j_error, exec_time, joint_tracking_error)
                    
#                     self.execute_trajectory(hover_joints, duration_sec=2.5)

#             print("\n[INFO] Experiment Complete. Returning to HOME...")
#             self.execute_trajectory(self.HOME_POSE, duration_sec=3.0)

#         except KeyboardInterrupt:
#             print("\n🚨 EMERGENCY STOP TRIGGERED! Returning to HOME safely...")
#             safe_up = [self.actual_joints[0], -1.2, 0.0, 0.0, 0.0]
#             self.execute_trajectory(safe_up, duration_sec=2.0)
#             self.execute_trajectory(self.HOME_POSE, duration_sec=3.0)
#         finally:
#             self.save_results()

#     def log_data(self, index, row, iteration, status, failure_type, 
#                  actual_j=None, actual_xyz=None, actual_q=None, 
#                  cartesian_err=np.nan, angular_err=np.nan, joint_err=np.nan,
#                  exec_time=np.nan, tracking_err=np.nan):
        
#         self.execution_results.append({
#             'execution_order': index, 
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
            
#             'cartesian_error_mm': cartesian_err,
#             'orientation_error_deg': angular_err,
#             'mean_joint_error_rad': joint_err,
#             'joint_tracking_error_rad': tracking_err, 
#             'execution_time_sec': exec_time           
#         })

#     def save_results(self):
#         if len(self.execution_results) == 0:
#             print("❌ No executions to save.")
#             return
            
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         csv_file = f"hardware_validation_raw_data_{timestamp}.csv"
#         txt_file = f"hardware_validation_metrics_{timestamp}.txt"
        
#         results_df = pd.DataFrame(self.execution_results)
#         results_df.to_csv(csv_file, index=False)
        
#         success_df = results_df[results_df['status'] == 'success']
#         inacc_df = results_df[results_df['status'] == 'inaccurate']
#         valid_df = pd.concat([success_df, inacc_df]) 
#         failed_df = results_df[results_df['status'] == 'failed']
        
#         total_attempts = len(results_df)
#         success_rate = (len(success_df) / total_attempts) * 100 if total_attempts > 0 else 0
        
#         # ✅ IK Feasibility / Sim-to-Real Disconnect Metric
#         planning_failures = len(failed_df[failed_df['failure_type'].isin(['safety_block_collision', 'sv_timeout', 'fk_failed'])])
#         ik_feasibility_rate = ((total_attempts - planning_failures) / total_attempts) * 100 if total_attempts > 0 else 0
        
#         # ✅ Orientation Constraint Satisfaction
#         if not valid_df.empty:
#             orientation_success_rate = (valid_df['orientation_error_deg'] < self.SUCCESS_ORIENT_THRESH).mean() * 100
#         else:
#             orientation_success_rate = 0.0
            
#         # ✅ Boundary vs Interior Regional Comparison
#         bound_df = valid_df[valid_df['test_type'].str.contains('Boundary')]
#         int_df = valid_df[valid_df['test_type'].str.contains('Interior')]
        
#         bound_err = bound_df['cartesian_error_mm'].mean() if not bound_df.empty else np.nan
#         int_err = int_df['cartesian_error_mm'].mean() if not int_df.empty else np.nan
        
#         bound_succ = (len(success_df[success_df['test_type'].str.contains('Boundary')]) / len(results_df[results_df['test_type'].str.contains('Boundary')])) * 100 if len(results_df[results_df['test_type'].str.contains('Boundary')]) > 0 else np.nan
#         int_succ = (len(success_df[success_df['test_type'].str.contains('Interior')]) / len(results_df[results_df['test_type'].str.contains('Interior')])) * 100 if len(results_df[results_df['test_type'].str.contains('Interior')]) > 0 else np.nan
        
#         avg_cartesian = valid_df['cartesian_error_mm'].mean() if not valid_df.empty else np.nan
#         max_cartesian = valid_df['cartesian_error_mm'].max() if not valid_df.empty else np.nan
        
#         avg_angular = valid_df['orientation_error_deg'].mean() if not valid_df.empty else np.nan
#         avg_time = valid_df['execution_time_sec'].mean() if not valid_df.empty else np.nan
        
#         grouped = valid_df.groupby('pose_id')['cartesian_error_mm']
#         std_dev_mean = grouped.std().mean() if not valid_df.empty else np.nan
        
#         # ✅ Explicit Sim-to-Real Terminology in Output
#         summary_text = (
#             f"============================================================\n"
#             f" 🎯 REAL-ROBOT HARDWARE VALIDATION METRICS (RESEARCH FORMAT)\n"
#             f"============================================================\n"
#             f"Total Target Executions:        {total_attempts}\n"
#             f"Sim-to-Real Feasibility Rate:   {ik_feasibility_rate:.2f}% (Passed Planning)\n"
#             f"Strict Execution Success Rate:  {success_rate:.2f}% (Error < 5mm)\n"
#             f"Orientation Constraint Success: {orientation_success_rate:.2f}% (Error < 5°)\n"
#             f"------------------------------------------------------------\n"
#             f"📏 SIM-TO-REAL CARTESIAN GAP (Accuracy)\n"
#             f"------------------------------------------------------------\n"
#             f"Mean Euclidean Gap:             {avg_cartesian:7.4f} mm\n"
#             f"Max Euclidean Gap:              {max_cartesian:7.4f} mm\n"
#             f"------------------------------------------------------------\n"
#             f"🗺 REGIONAL CAPABILITY (Boundary vs Interior)\n"
#             f"------------------------------------------------------------\n"
#             f"Boundary Mean Error:            {bound_err:7.4f} mm\n"
#             f"Interior Mean Error:            {int_err:7.4f} mm\n"
#             f"Boundary Success Rate:          {bound_succ:7.2f} %\n"
#             f"Interior Success Rate:          {int_succ:7.2f} %\n"
#             f"------------------------------------------------------------\n"
#             f"🔄 HARDWARE REPEATABILITY (Precision)\n"
#             f"------------------------------------------------------------\n"
#             f"Mean Spatial Std Dev:           ±{std_dev_mean:6.4f} mm\n"
#             f"------------------------------------------------------------\n"
#             f"📐 6-DOF ORIENTATION FIDELITY\n"
#             f"------------------------------------------------------------\n"
#             f"Mean Angular Error:             {avg_angular:7.4f}°\n"
#             f"------------------------------------------------------------\n"
#             f"⏱ KINEMATIC PERFORMANCE\n"
#             f"------------------------------------------------------------\n"
#             f"Average Trajectory Exec Time:   {avg_time:7.3f} sec\n"
#             f"============================================================\n"
#         )
        
#         with open(txt_file, "w") as f:
#             f.write(summary_text)

#         print(summary_text)
#         print(f"💾 Saved Raw Execution Data to: {csv_file}")
#         print(f"📄 Saved Results Summary to:    {txt_file}")
        
#         if not valid_df.empty:
#             self.generate_publication_plot(valid_df, timestamp)

#     def generate_publication_plot(self, df, timestamp):
#         try:
#             plt.figure(figsize=(10, 6))
#             types = df['test_type'].unique()
#             data_to_plot = [df[df['test_type'] == t]['cartesian_error_mm'].dropna() for t in types]
#             box = plt.boxplot(data_to_plot, labels=types, patch_artist=True)
#             colors =['#4C72B0', '#DD8452'] 
#             for patch, color in zip(box['boxes'], colors):
#                 patch.set_facecolor(color)
#                 patch.set_alpha(0.7)
#             plt.title('Sim-to-Real Validation: Positioning Error by Region', fontsize=14, fontweight='bold')
#             plt.ylabel('Sim-to-Real Cartesian Gap (mm)', fontsize=12, fontweight='bold')
#             plt.grid(axis='y', linestyle='--', alpha=0.7)
#             plot_file = f"hardware_error_plot_{timestamp}.png"
#             plt.savefig(plot_file, dpi=300, bbox_inches='tight')
#             print(f"📈 Saved Paper-Ready Plot:      {plot_file}\n")
#         except Exception: pass

# def main(args=None):
#     rclpy.init(args=args)
#     input_csv = "real_robot_40_test_poses_table_top.csv" 
#     executor = RealRobotHardwareValidator(input_csv)
#     executor.run_experiment()
#     executor.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()