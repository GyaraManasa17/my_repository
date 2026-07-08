import rclpy
from scene_manager import SceneManager

def main():
    rclpy.init()
    node = rclpy.create_node('test_scene_node')
    
    # NOTE: Make sure "link1" matches the base_link from your parser
    manager = SceneManager(node, base_frame="link1") 
    
    print("Spawning obstacles...")
    manager.setup_obstacle_scene()
    print("✅ Done! Look at RViz. Is the robot touching the green boxes?")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()