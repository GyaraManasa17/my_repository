from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
import os

def generate_launch_description():
    # IMPORTANT: Change this to your 5DOF or 6DOF moveit config package accordingly
    moveit_config = MoveItConfigsBuilder("open_manipulator_x").to_moveit_configs()

    # Path to the YAML file you just created
    benchmark_yaml = "/home/ubuntu/metric_ws/benchmark_configs/benchmark_5dof.yaml" 
    # Change to benchmark_6dof.yaml when testing the 6DOF arm

    benchmark_node = Node(
        package="moveit_ros_benchmarks",
        executable="moveit_run_benchmark",
        name="moveit_run_benchmark",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            benchmark_yaml,
        ],
    )

    return LaunchDescription([benchmark_node])