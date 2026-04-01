#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sys
import argparse

# Import benchmark modules
from m1_robot_parser import RobotKinematicParser
from m2_sampling_fk import WorkspaceSampler
from m3_workspace_metrics import WorkspaceAnalyzer
from m4_dexterity_metrics import DexterityAnalyzer
from m5_ik_metrics import IKPerformanceAnalyzer
from m6_planning_metrics import PlanningPerformanceAnalyzer
from m7_trajectory_metrics import TrajectoryPerformanceAnalyzer

# IMPORT MISSING CLASS
from planner_interface import PlannerInterface


class BenchmarkRunner(Node):

    def __init__(self, urdf_path=None, srdf_path=None):

        super().__init__('benchmark_runner_node')

        self.get_logger().info("🚀 Initializing Robotic Arm Analysis Benchmark...")

        self.urdf_path = urdf_path
        self.srdf_path = srdf_path

        self.parser = None
        self.sampler = None
        self.planner_interface = None

    def run_benchmark(self):

        try:

            # ==========================================
            # MODULE 1: URDF / SRDF PARSER
            # ==========================================

            self.get_logger().info("--- [STARTING MODULE 1: URDF/SRDF Parsing] ---")

            self.parser = RobotKinematicParser(
                self,
                urdf_path=self.urdf_path,
                srdf_path=self.srdf_path
            )

            self.parser.load_and_parse(timeout_sec=10.0)

            self.get_logger().info("✅ Module 1 completed successfully.\n")

            # ==========================================
            # MODULE 2: WORKSPACE SAMPLING (FK)
            # ==========================================

            self.get_logger().info("--- [STARTING MODULE 2: Workspace Sampling (FK)] ---")

            self.sampler = WorkspaceSampler(self, self.parser)

            n_samples = 10000
            seed = 42

            self.get_logger().info(f"⚙️ Sampling {n_samples} workspace points...")

            self.sampler.sample_workspace(
                n_samples=n_samples,
                seed=seed
            )

            self.get_logger().info("✅ Module 2 completed successfully.\n")

            # ==========================================
            # MODULE 3: WORKSPACE METRICS
            # ==========================================

            self.get_logger().info("--- [MODULE 3: Workspace Analysis] ---")

            analyzer = WorkspaceAnalyzer(self)

            metrics = analyzer.analyze()

            if metrics:
                self.get_logger().info("✅ Module 3 completed successfully.")
            else:
                self.get_logger().warning("⚠️ Module 3 finished but no metrics generated.")

            # ==========================================
            # MODULE 4 : DEXTERITY ANALYSIS
            # ==========================================

            try:

                self.get_logger().info("🧠 Starting Module 4: Dexterity Analysis...")

                dexterity_analyzer = DexterityAnalyzer(
                    node=self,
                    sampler=self.sampler,
                    parser=self.parser
                )

                dex_metrics = dexterity_analyzer.analyze()

                if dex_metrics:

                    self.get_logger().info("✅ Dexterity Analysis Completed.")

                    self.get_logger().info(
                        f"📊 Mean Manipulability: {dex_metrics['manipulability_mean']:.4f}"
                    )

                    self.get_logger().info(
                        f"📊 Global Dexterity Index: {dex_metrics['gdi']:.4f}"
                    )

                    self.get_logger().info(
                        f"📊 Singularity Ratio: {dex_metrics['singularity_ratio']:.4f}"
                    )

                else:

                    self.get_logger().warning("⚠️ Dexterity analysis returned no results.")

            except Exception as e:

                self.get_logger().error(f"❌ Module 4 failed: {str(e)}")

            # ==========================================
            # MODULE 5 : IK PERFORMANCE
            # ==========================================

            try:

                self.get_logger().info("🔧 Starting Module 5: IK Performance Analysis...")

                # ik_analyzer = IKPerformanceAnalyzer(
                #     node=self,
                #     sampler=self.sampler
                # )
                # Pass the group_name and base_link from the parser!
                ik_analyzer = IKPerformanceAnalyzer(
                    node=self,
                    group_name=self.parser.group_name,
                    base_link=self.parser.base_link
                )

                ik_metrics = ik_analyzer.analyze()

                if ik_metrics:

                    self.get_logger().info("✅ Module 5 completed successfully.")

                    self.get_logger().info(
                        f"📊 IK Success Rate: {ik_metrics['ik_success_rate']:.4f}"
                    )

                    self.get_logger().info(
                        f"📊 Mean IK Time: {ik_metrics['ik_mean_time']:.6f}"
                    )

                    self.get_logger().info(
                        f"📊 IK Singularity Rate: {ik_metrics['ik_singularity_rate']:.4f}"
                    )

                else:

                    self.get_logger().warning("⚠️ Module 5 finished but produced no metrics.")

            except Exception as e:

                self.get_logger().error(f"❌ Module 5 failed: {str(e)}")

            # ==========================================
            # MODULE 6 : MOTION PLANNING PERFORMANCE
            # ==========================================

            self.get_logger().info("")
            self.get_logger().info("=======================================================")
            self.get_logger().info("MODULE 6: MOTION PLANNING PERFORMANCE BENCHMARK")
            self.get_logger().info("=======================================================")

            try:

                # STORE planner interface in class
                # self.planner_interface = PlannerInterface(self)
                # Pass the robot parameters from the parser to the planner interface!
                self.planner_interface = PlannerInterface(
                    node=self,
                    group_name=self.parser.group_name,
                    base_link=self.parser.base_link,
                    end_link=self.parser.end_link,
                    joints=self.parser.joints
                )

                planner_name = "RRTConnect"

                planning_analyzer = PlanningPerformanceAnalyzer(
                    node=self,
                    planner_interface=self.planner_interface,
                    planner_name=planner_name
                )

                planning_metrics = planning_analyzer.analyze(max_samples=1000)

                if planning_metrics is None:
                    self.get_logger().error("❌ Module 6 failed.")
                else:
                    self.get_logger().info("✅ Module 6 completed successfully.")

            except Exception as e:

                self.get_logger().error(f"❌ Module 6 error: {str(e)}")

            # ==========================================
            # MODULE 7 : TRAJECTORY QUALITY
            # ==========================================

            self.get_logger().info("--- [STARTING MODULE 7: Trajectory Quality Analysis] ---")

            try:

                traj_analyzer = TrajectoryPerformanceAnalyzer(
                    node=self,
                    planner_interface=self.planner_interface
                )

                metrics_traj = traj_analyzer.analyze()

                if metrics_traj:

                    self.get_logger().info(
                        "✅ Module 7 completed successfully. Trajectory metrics and plots saved!\n"
                    )

                else:

                    self.get_logger().warning(
                        "⚠️ Module 7 finished but no valid trajectories were analyzed.\n"
                    )

            except Exception as e:

                self.get_logger().error(f"❌ Module 7 failed: {str(e)}\n")

        except Exception as e:

            self.get_logger().error(f"❌ Benchmark aborted due to an error: {e}")
            sys.exit(1)


def main(args=None):

    parser = argparse.ArgumentParser(description="Robotic Arm Benchmark Runner")

    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to URDF or XACRO file"
    )

    parser.add_argument(
        "--srdf",
        type=str,
        default=None,
        help="Path to SRDF file"
    )

    parsed_args = parser.parse_args()

    rclpy.init(args=args)

    runner = BenchmarkRunner(
        urdf_path=parsed_args.urdf,
        srdf_path=parsed_args.srdf
    )

    try:

        runner.run_benchmark()

    except KeyboardInterrupt:

        runner.get_logger().info("🛑 Benchmark interrupted by user.")

    finally:

        runner.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()