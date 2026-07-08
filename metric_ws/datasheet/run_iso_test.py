import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros
import pandas as pd
import time
import json
import os

class ISOTestRunner(Node):
    def __init__(self, csv_file):
        super().__init__('iso_test_runner')
        
        # --- CRITICAL FIX FOR GAZEBO ---
        # Forces the node to use simulation time to fix the "1.77e9 delay" TF error
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        
        # --- UPDATED CONFIGURATION ---
        self.action_topic = '/arm_controller/follow_joint_trajectory' # Updated from your 'ros2 action list'
        self.base_frame = 'link1'           # Confirmed from your tf2_monitor
        self.ee_frame = 'end_effector_link' # Confirmed from your tf2_monitor
        self.cycles = 10                    # ISO standard recommends 30 cycles
        
        # Load the 5 poses
        self.df = pd.read_csv(csv_file)
        self.joint_names =['joint1', 'joint2', 'joint3', 'joint4', 'joint5_roll']
        
        # Setup Action Client to move the robot
        self.action_client = ActionClient(self, FollowJointTrajectory, self.action_topic)
        
        # Setup TF Listener to measure actual position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.recorded_data = {f"P{i+1}":[] for i in range(5)}
        self.target_data = {f"P{i+1}": [row['x'], row['y'], row['z']] for i, row in self.df.iterrows()}

    def get_actual_pose(self):
        """Reads the physical (or simulated) end-effector position from TF2"""
        try:
            # Query the latest available transform (Time=0) and wait up to 3 seconds for it
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, 
                self.ee_frame, 
                rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=3.0)
            )
            return[trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
        except Exception as e:
            self.get_logger().error(f"TF Error: {e}")
            return None

    def move_to_joints(self, joint_values):
        """Sends an action goal to the robot to move to the specified joints"""
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Action server '{self.action_topic}' not available! Is Gazebo running?")
            return False

        goal_msg = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = list(joint_values)
        point.time_from_start.sec = 3  # Give the robot 3 seconds to reach the pose
        trajectory.points =[point]
        goal_msg.trajectory = trajectory

        # Send command asynchronously
        future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected by the action server.")
            return False

        # Wait for the robot to finish moving
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True

    def run_test(self):
        self.get_logger().info(f"Starting ISO 9283 Test: {self.cycles} cycles...")
        
        # Pause briefly to allow the TF buffer to fill up with Gazebo transforms
        time.sleep(2.0)
        
        for cycle in range(self.cycles):
            self.get_logger().info(f"\n--- CYCLE {cycle + 1}/{self.cycles} ---")
            
            for index, row in self.df.iterrows():
                pose_name = f"P{index+1}"
                joint_vals = [row[j] for j in self.joint_names]
                
                # 1. Command Robot
                self.get_logger().info(f"Moving to {pose_name}...")
                success = self.move_to_joints(joint_vals)
                
                if not success:
                    self.get_logger().warn(f"Failed to move to {pose_name}, skipping.")
                    continue
                
                # 2. Wait 1 second for the arm to settle physically in Gazebo
                time.sleep(4.0) 
                
                # 3. Measure Actual Pose from TF
                actual_pos = self.get_actual_pose()
                if actual_pos:
                    self.recorded_data[pose_name].append(actual_pos)
                    self.get_logger().info(f"{pose_name} Reached: {actual_pos}")
                    
        # 4. Save results
        output_file = 'iso_test_results.json'
        with open(output_file, 'w') as f:
            json.dump({"targets": self.target_data, "achieved": self.recorded_data}, f, indent=4)
        self.get_logger().info(f"✅ Test Complete! Data saved to '{output_file}'")

def main(args=None):
    rclpy.init(args=args)
    
    # Ensure this matches the CSV generated in the previous step
    csv_file = "iso_9283_test_poses.csv" 
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: '{csv_file}' not found.")
        print("Please ensure the CSV is in the same directory where you are running this script.")
        return

    runner = ISOTestRunner(csv_file)
    runner.run_test()
    
    runner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient
# from control_msgs.action import FollowJointTrajectory
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# import tf2_ros
# import pandas as pd
# import time
# import json
# import os

# class ISOTestRunner(Node):
#     def __init__(self, csv_file):
#         super().__init__('iso_test_runner')
        
#         # --- CONFIGURATION ---
#         # CHANGE THIS if your action server is named differently (e.g., /arm_controller/follow_joint_trajectory)
#         self.action_topic = '/joint_trajectory_controller/follow_joint_trajectory'
#         self.base_frame = 'link1'  # Base link of OpenManipulator
#         self.ee_frame = 'end_effector_link' # End effector link
#         self.cycles = 30 # ISO standard recommends 30 cycles
        
#         # Load the 5 poses
#         self.df = pd.read_csv(csv_file)
#         self.joint_names =['joint1', 'joint2', 'joint3', 'joint4', 'joint5_roll']
        
#         # Setup Action Client to move the robot
#         self.action_client = ActionClient(self, FollowJointTrajectory, self.action_topic)
        
#         # Setup TF Listener to measure actual position
#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
#         self.recorded_data = {f"P{i+1}":[] for i in range(5)}
#         self.target_data = {f"P{i+1}": [row['x'], row['y'], row['z']] for i, row in self.df.iterrows()}

#     def get_actual_pose(self):
#         """Reads the physical (or simulated) end-effector position from TF2"""
#         try:
#             trans = self.tf_buffer.lookup_transform(
#                 self.base_frame, self.ee_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0))
#             return[trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
#         except Exception as e:
#             self.get_logger().error(f"TF Error: {e}")
#             return None

#     def move_to_joints(self, joint_values):
#         """Sends an action goal to the robot to move to the specified joints"""
#         if not self.action_client.wait_for_server(timeout_sec=5.0):
#             self.get_logger().error("Action server not available!")
#             return False

#         goal_msg = FollowJointTrajectory.Goal()
#         trajectory = JointTrajectory()
#         trajectory.joint_names = self.joint_names
        
#         point = JointTrajectoryPoint()
#         point.positions = list(joint_values)
#         point.time_from_start.sec = 3  # Give the robot 3 seconds to reach the pose
#         trajectory.points = [point]
#         goal_msg.trajectory = trajectory

#         future = self.action_client.send_goal_async(goal_msg)
#         rclpy.spin_until_future_complete(self, future)
#         goal_handle = future.result()
        
#         if not goal_handle.accepted:
#             return False

#         result_future = goal_handle.get_result_async()
#         rclpy.spin_until_future_complete(self, result_future)
#         return True

#     def run_test(self):
#         self.get_logger().info(f"Starting ISO 9283 Test: {self.cycles} cycles...")
        
#         for cycle in range(self.cycles):
#             self.get_logger().info(f"\n--- CYCLE {cycle + 1}/{self.cycles} ---")
            
#             for index, row in self.df.iterrows():
#                 pose_name = f"P{index+1}"
#                 joint_vals = [row[j] for j in self.joint_names]
                
#                 # 1. Command Robot
#                 self.get_logger().info(f"Moving to {pose_name}...")
#                 self.move_to_joints(joint_vals)
                
#                 # 2. Wait for settling
#                 time.sleep(1.0) 
                
#                 # 3. Measure Actual Pose
#                 actual_pos = self.get_actual_pose()
#                 if actual_pos:
#                     self.recorded_data[pose_name].append(actual_pos)
#                     self.get_logger().info(f"{pose_name} Reached: {actual_pos}")
                    
#         # 4. Save results
#         with open('iso_test_results.json', 'w') as f:
#             json.dump({"targets": self.target_data, "achieved": self.recorded_data}, f, indent=4)
#         self.get_logger().info("✅ Test Complete! Data saved to 'iso_test_results.json'")

# def main(args=None):
#     rclpy.init(args=args)
#     # REPLACE with the name of the CSV file you just generated
#     csv_file = "iso_9283_test_poses.csv" 
    
#     if not os.path.exists(csv_file):
#         print(f"Error: {csv_file} not found.")
#         return

#     runner = ISOTestRunner(csv_file)
#     runner.run_test()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()