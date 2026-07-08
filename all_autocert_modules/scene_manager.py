import rclpy
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, PlanningSceneWorld, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

class SceneManager:
    def __init__(self, node, base_frame="base_link"):
        self.node = node
        self.base_frame = base_frame
        
        self.node.get_logger().info("⏳ Waiting for MoveIt '/apply_planning_scene' service...")
        self.scene_client = self.node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        
        if not self.scene_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error("❌ Scene service not found! Obstacles cannot be added.")
        else:
            self.node.get_logger().info("✅ Scene Manager Connected!")

    def _send_scene_request(self, collision_objects):
        req = ApplyPlanningScene.Request()
        scene = PlanningScene()
        scene.is_diff = True
        scene.world = PlanningSceneWorld()
        scene.world.collision_objects = collision_objects
        req.scene = scene

        future = self.scene_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
        return future.result().success if future.done() else False

    def clear_scene(self):
        self.node.get_logger().info("🧹 Clearing Planning Scene (Empty Void Mode)...")
        obj_table = CollisionObject()
        obj_table.id = "table"
        obj_table.operation = CollisionObject.REMOVE
        
        obj_wall = CollisionObject()
        obj_wall.id = "obstacle_wall"
        obj_wall.operation = CollisionObject.REMOVE

        self._send_scene_request([obj_table, obj_wall])

    def setup_obstacle_scene(self):
        self.node.get_logger().info("🧱 Setting up Obstacle Scene...")
        objects_to_add = []

        # 1. ADD TABLE (Floor)
        table = CollisionObject()
        table.id = "table"
        table.header.frame_id = self.base_frame
        table.operation = CollisionObject.ADD
        
        table_shape = SolidPrimitive()
        table_shape.type = SolidPrimitive.BOX
        table_shape.dimensions = [2.0, 2.0, 0.05] 
        
        table_pose = Pose()
        table_pose.position.x = 0.0
        table_pose.position.y = 0.0
        table_pose.position.z = -0.10 # Dropped lower so link1 is completely safe
        
        table.primitives.append(table_shape)
        table.primitive_poses.append(table_pose)
        objects_to_add.append(table)

        # 2. ADD WALL (Moved to the SIDE!)
        wall = CollisionObject()
        wall.id = "obstacle_wall"
        wall.header.frame_id = self.base_frame
        wall.operation = CollisionObject.ADD
        
        wall_shape = SolidPrimitive()
        wall_shape.type = SolidPrimitive.BOX
        wall_shape.dimensions = [0.25, 0.05, 0.30] # 25cm long (X), 5cm thick (Y), 30cm tall (Z)
        
        wall_pose = Pose()
        wall_pose.position.x = 0.15  # Slightly forward
        wall_pose.position.y = 0.15  # Shifted 15cm to the LEFT (Green arrow in RViz)
        wall_pose.position.z = 0.05  # Sitting on the table
        
        wall.primitives.append(wall_shape)
        wall.primitive_poses.append(wall_pose)
        objects_to_add.append(wall)

        success = self._send_scene_request(objects_to_add)
        if success:
            self.node.get_logger().info("✅ Obstacles successfully added to MoveIt! Robot is clear of collisions.")
        else:
            self.node.get_logger().error("❌ Failed to add obstacles.")