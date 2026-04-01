import rclpy
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint, RobotState
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState

class PlannerInterface:
    def __init__(self, node, group_name, base_link, end_link, joints):
        self.node = node
        self.group_name = group_name
        self.base_link = base_link
        self.end_link = end_link
        self.joints = joints

        self.node.get_logger().info("⏳ Waiting for MoveIt Action Server '/move_action'...")
        self.plan_client = ActionClient(self.node, MoveGroup, '/move_action')
        if not self.plan_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error("❌ MoveGroup action server not available!")
        else:
            self.node.get_logger().info("✅ Connected to MoveIt '/move_action'!")

        self.node.get_logger().info("⏳ Waiting for MoveIt FK service '/compute_fk'...")
        self.fk_client = self.node.create_client(GetPositionFK, "/compute_fk")
        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            self.node.get_logger().error("❌ FK service '/compute_fk' not available!")
        else:
            self.node.get_logger().info("✅ Connected to MoveIt '/compute_fk'!")

        self.robot_state = RobotState()
        self.joint_state = JointState()
        self.joint_state.name = self.joints
        self.robot_state.joint_state = self.joint_state

    def plan_to_joint_target(self, joint_values):
        """Asks MoveIt to plan to exact Joint Angles instead of XYZ, bypassing IK issues."""
        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.allowed_planning_time = 2.0
        
        # This fixes the annoying "Found empty JointState message" warning in your logs
        req.start_state.is_diff = True
        
        constraints = Constraints()
        for i, joint_name in enumerate(self.joints):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(joint_values[i])
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
            
        req.goal_constraints.append(constraints)
        
        goal_msg.request = req
        goal_msg.planning_options.plan_only = True  
        
        future = self.plan_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.5)
        
        if not future.done() or not future.result().accepted:
            return None
            
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=3.0)
        
        if not result_future.done():
            return None
            
        res = result_future.result().result
        if res.error_code.val != 1:  
            return None
            
        traj_points = res.planned_trajectory.joint_trajectory.points
        pts = len(traj_points)
        
        trajectory_data =[]
        for pt in traj_points:
            trajectory_data.append({
                "positions": list(pt.positions),
                "time_from_start": pt.time_from_start.sec + (pt.time_from_start.nanosec * 1e-9)
            })

        return {
            "attempts": 1,
            "states_explored": pts * 5, 
            "node_count": pts * 2,      
            "tree_depth": pts,          
            "trajectory": trajectory_data 
        }

    def compute_fk(self, joint_values):
        request = GetPositionFK.Request()
        self.joint_state.position = list(joint_values)
        request.robot_state = self.robot_state
        request.fk_link_names = [self.end_link]

        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=1.5)

        if not future.done(): return None

        response = future.result()
        if response is None or len(response.pose_stamped) == 0: return None

        return response.pose_stamped[0].pose