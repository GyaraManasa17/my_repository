#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

class SceneManager(Node):
    def __init__(self):
        super().__init__('scene_manager')
        self.client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /apply_planning_scene...")

    def reset_scene(self):
        """FIX 1: Explicitly REMOVE known objects"""
        scene = PlanningScene()
        scene.is_diff = True
        
        # Remove wall if exists
        remove_wall = CollisionObject()
        remove_wall.id = "wall_box"
        remove_wall.operation = CollisionObject.REMOVE
        scene.world.collision_objects = [remove_wall]
        
        req = ApplyPlanningScene.Request(scene=scene)
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def create_roll_critical_scene(self):
        self.reset_scene()
        
        # Add wall
        collision = CollisionObject()
        collision.id = "wall_box"
        collision.header = Header(frame_id="world")
        
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.02, 0.3, 0.3]
        
        pose = Pose()
        pose.position.x = 0.18
        pose.position.y = 0.05
        pose.position.z = 0.15
        collision.primitives.append(primitive)
        collision.primitive_poses.append(pose)
        collision.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [collision]
        
        req = ApplyPlanningScene.Request(scene=scene)
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info("✅ Scene RESET + wall added")

def main():
    rclpy.init()
    node = SceneManager()
    node.create_roll_critical_scene()
    rclpy.shutdown()
if __name__ == '__main__': main()






# #!/usr/bin/env python3
# """
# Scene Builder for Constrained Benchmark Environment
# Creates obstacles that force the robot to use wrist roll
# """
# import rclpy
# from rclpy.node import Node
# from moveit_msgs.srv import ApplyPlanningScene
# from moveit_msgs.msg import PlanningScene, CollisionObject
# from shape_msgs.msg import SolidPrimitive
# from geometry_msgs.msg import Pose
# import time


# class SceneBuilder(Node):
#     def __init__(self):
#         super().__init__('scene_builder')
        
#         # Client for applying planning scene
#         self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        
#         self.get_logger().info('Waiting for apply_planning_scene service...')
#         while not self.scene_client.wait_for_service(timeout_sec=10.0):
#             self.get_logger().info('Still waiting...')
        
#         self.get_logger().info('✓ Scene builder connected')
    
#     def create_box(self, name, dimensions, position):
#         """Create a collision box"""
#         obj = CollisionObject()
#         obj.id = name
#         obj.header.frame_id = "world"
#         obj.operation = CollisionObject.ADD
        
#         box = SolidPrimitive()
#         box.type = SolidPrimitive.BOX
#         box.dimensions = dimensions  # [x, y, z]
        
#         pose = Pose()
#         pose.position.x = position[0]
#         pose.position.y = position[1]
#         pose.position.z = position[2]
#         pose.orientation.w = 1.0
        
#         obj.primitives.append(box)
#         obj.primitive_poses.append(pose)
        
#         return obj
    
#     def build_constrained_scene(self):
#         """Build a scene that REQUIRES wrist roll to succeed"""
#         scene = PlanningScene()
#         scene.is_diff = True
        
#         # Back wall - move slightly forward to give more room
#         # scene.world.collision_objects.append(
#         #     self.create_box(
#         #         'back_wall',
#         #         [0.05, 0.5, 0.4],
#         #         [0.12, 0.0, 0.2]  # Moved forward a tiny bit (was 0.1)
#         #     )
#         # )
        
#         # Left/Right walls - create a narrow corridor
#         # Width between walls: 0.24m (from -0.12 to +0.12)
#         # Gripper width is about 0.1m, so this forces orientation change
#         scene.world.collision_objects.append(
#             self.create_box(
#                 'left_wall',
#                 [0.15, 0.14, 0.4],  # Made narrower in y-direction
#                 [0.3, -0.12, 0.2]
#             )
#         )
        
#         scene.world.collision_objects.append(
#             self.create_box(
#                 'right_wall',
#                 [0.15, 0.14, 0.4],
#                 [0.3, 0.12, 0.2]
#             )
#         )
        
#         # Top barrier - raised higher to force lower approach
#         scene.world.collision_objects.append(
#             self.create_box(
#                 'top_barrier',
#                 [0.25, 0.3, 0.05],
#                 [0.4, 0.0, 0.28]  # Raised higher
#             )
#         )
        
#         # Bottom plate
#         scene.world.collision_objects.append(
#             self.create_box(
#                 'bottom_plate',
#                 [0.5, 0.5, 0.02],
#                 [0.3, 0.0, -0.01]
#             )
#         )
        
#         return scene
    
#     def apply_scene(self):
#         """Apply the scene to MoveIt"""
#         self.get_logger().info('Building constrained benchmark scene...')
#         scene = self.build_constrained_scene()
        
#         request = ApplyPlanningScene.Request()
#         request.scene = scene
        
#         future = self.scene_client.call_async(request)
#         rclpy.spin_until_future_complete(self, future)
        
#         if future.result() is not None:
#             self.get_logger().info('✓ Scene applied successfully!')
#             time.sleep(2.0)  # Give time for scene to propagate
#             self.get_logger().info('')
#             self.get_logger().info('Scene objects:')
#             self.get_logger().info('  - Left wall: (0.3, -0.15, 0.2) [0.1x0.2x0.4]')
#             self.get_logger().info('  - Right wall: (0.3, 0.15, 0.2) [0.1x0.2x0.4]')
#             self.get_logger().info('  - Back wall: (0.15, 0.0, 0.2) [0.05x0.5x0.4]')
#             self.get_logger().info('  - Top barrier: (0.35, 0.0, 0.3) [0.2x0.3x0.05]')
#             self.get_logger().info('  - Bottom plate: (0.3, 0.0, -0.01) [0.5x0.5x0.02]')
#             self.get_logger().info('')
#             self.get_logger().info('Target pose: (0.35, 0.0, 0.15) pointing down')
#             return True
#         else:
#             self.get_logger().error('✗ Failed to apply scene')
#             return False
    
#     def verify_scene(self):
#         """Print verification commands"""
#         self.get_logger().info('')
#         self.get_logger().info('To verify scene, run:')
#         self.get_logger().info('  ros2 topic echo /planning_scene --once | grep -A 10 "collision_objects"')
#         self.get_logger().info('')


# def main(args=None):
#     rclpy.init(args=args)
    
#     node = SceneBuilder()
#     success = node.apply_scene()
    
#     if success:
#         time.sleep(2.0)
#         node.verify_scene()
    
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()