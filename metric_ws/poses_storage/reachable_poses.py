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
import sys
from datetime import datetime

from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
import alphashape


class AdvancedWorkspaceAnalyzer(Node):

    def __init__(self, dof):

        super().__init__("advanced_workspace_analyzer")

        self.dof = dof
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        self.end_link = "end_effector_link"

        # --------------------------------------------------
        # JOINT CONFIGURATION
        # --------------------------------------------------

        if dof == 5:

            self.joints = ["joint1", "joint2", "joint3", "joint4"]

            self.limits = [
                (-math.pi, math.pi),
                (-1.5, 1.5),
                (-1.5, 1.4),
                (-1.7, 1.97)
            ]

            self.tag = "_5dof"

        elif dof == 6:

            self.joints = ["joint1", "joint2", "joint3", "joint4", "joint5_roll"]

            self.limits = [
                (-math.pi, math.pi),
                (-1.5, 1.5),
                (-1.5, 1.4),
                (-1.7, 1.97),
                (-math.pi, math.pi)
            ]

            self.tag = "_6dof"

        else:

            raise ValueError("DOF must be 5 or 6")

    # --------------------------------------------------

    def wait_for_services(self):

        self.get_logger().info("Waiting for MoveIt FK service...")

        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            return False

        return True

    # --------------------------------------------------

    def run_advanced_analysis(self, num_samples=10000):

        if not self.wait_for_services():

            self.get_logger().error("MoveIt FK service not available!")
            return

        print("\n" + "="*80)
        print(f"🚀 WORKSPACE ANALYSIS ({self.dof} DOF)")
        print("="*80)

        # --------------------------------------------------
        # REPRODUCIBLE RANDOM SAMPLING
        # --------------------------------------------------

        np.random.seed(42)

        num_joints = len(self.joints)

        samples = np.zeros((num_samples, num_joints))

        for j in range(num_joints):

            min_val, max_val = self.limits[j]

            samples[:, j] = np.random.uniform(min_val, max_val, num_samples)

        results = []

        # --------------------------------------------------
        # FK REQUEST SETUP
        # --------------------------------------------------

        req = GetPositionFK.Request()
        req.header.frame_id = "link1"
        req.fk_link_names = [self.end_link]

        robot_state = RobotState()
        joint_state = JointState(name=self.joints)

        start_time = time.time()

        # --------------------------------------------------
        # FK LOOP
        # --------------------------------------------------

        for i in range(num_samples):

            if i % 1000 == 0 and i > 0:

                rate = i / (time.time() - start_time)

                print(f"⏳ {i}/{num_samples} poses processed ({rate:.1f} poses/sec)")

            joint_state.position = [float(v) for v in samples[i]]

            robot_state.joint_state = joint_state
            req.robot_state = robot_state

            future = self.fk_client.call_async(req)

            rclpy.spin_until_future_complete(self, future)

            resp = future.result()

            if resp and resp.error_code.val == resp.error_code.SUCCESS:

                pose = resp.pose_stamped[0].pose

                x = pose.position.x
                y = pose.position.y
                z = pose.position.z

                q = pose.orientation

                rot = R.from_quat([q.x, q.y, q.z, q.w])

                roll, pitch, yaw = rot.as_euler('xyz', degrees=False)

                results.append([
                    *samples[i],
                    x, y, z,
                    roll, pitch, yaw
                ])

        elapsed = time.time() - start_time

        print(f"\n✅ {len(results)} valid poses extracted in {elapsed:.2f} sec")

        self.process_results(results, num_samples)

    # --------------------------------------------------

    def process_results(self, results, num_samples):

        if not results:

            print("❌ No valid FK results")
            return

        data = np.array(results)

        joint_count = len(self.joints)

        points = data[:, joint_count:joint_count+3]

        valid_count = len(results)

        fk_success_rate = (valid_count / num_samples) * 100

        # --------------------------------------------------
        # BOUNDING BOX
        # --------------------------------------------------

        min_x, max_x = points[:,0].min(), points[:,0].max()
        min_y, max_y = points[:,1].min(), points[:,1].max()
        min_z, max_z = points[:,2].min(), points[:,2].max()

        max_reach = np.max(np.linalg.norm(points, axis=1))

        bbox_vol = (max_x-min_x)*(max_y-min_y)*(max_z-min_z)

        # --------------------------------------------------
        # CONVEX HULL
        # --------------------------------------------------

        try:

            print("⚙️ Computing Convex Hull")

            hull = ConvexHull(points)

            convex_volume = hull.volume
            convex_area = hull.area

        except Exception:

            convex_volume = 0
            convex_area = 0

        # --------------------------------------------------
        # ALPHA SHAPE
        # --------------------------------------------------

        try:

            print("⚙️ Computing Alpha Shape")

            alpha_val = 5.0

            alpha_mesh = alphashape.alphashape(points, alpha_val)

            alpha_volume = alpha_mesh.volume
            alpha_area = alpha_mesh.area

            alpha_vertices = np.array(alpha_mesh.vertices)

        except Exception:

            alpha_volume = 0
            alpha_area = 0
            alpha_vertices = []

        # --------------------------------------------------
        # DENSITY METRICS
        # --------------------------------------------------

        bbox_density = valid_count/bbox_vol if bbox_vol > 0 else 0
        alpha_density = valid_count/alpha_volume if alpha_volume > 0 else 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --------------------------------------------------
        # SAVE REACHABLE POSES (WITH JOINTS)
        # --------------------------------------------------

        joint_cols = self.joints
        pose_cols = ['x','y','z','roll','pitch','yaw']

        columns = joint_cols + pose_cols

        poses_df = pd.DataFrame(data, columns=columns)

        poses_file = f"omx_all_reachable_poses{self.tag}_{timestamp}.csv"

        poses_df.to_csv(poses_file, index=False)

        # --------------------------------------------------
        # SAVE ALPHA BOUNDARY
        # --------------------------------------------------

        if len(alpha_vertices) > 0:

            boundary_df = pd.DataFrame(alpha_vertices, columns=['x','y','z'])

            hull_file = f"omx_alpha_boundary{self.tag}_{timestamp}.csv"

            boundary_df.to_csv(hull_file, index=False)

        else:

            hull_file = "N/A"

        # --------------------------------------------------
        # SUMMARY FILE
        # --------------------------------------------------

        results_file = f"omx_results{self.tag}_{timestamp}.txt"

        summary = f"""
============================================================
TRUE WORKSPACE METRICS
============================================================

DOF: {self.dof}

Attempted Samples: {num_samples}
Valid FK Results: {valid_count}

FK Success Rate: {fk_success_rate:.2f} %

------------------------------------------------------------

X Range: [{min_x:.4f}, {max_x:.4f}]
Y Range: [{min_y:.4f}, {max_y:.4f}]
Z Range: [{min_z:.4f}, {max_z:.4f}]

Max Reach: {max_reach:.4f} m

Bounding Box Volume: {bbox_vol:.4f} m^3
Bounding Density: {bbox_density:.2f}

Convex Hull Volume: {convex_volume:.4f} m^3
Convex Surface Area: {convex_area:.4f}

Alpha Shape Volume: {alpha_volume:.4f} m^3
Alpha Surface Area: {alpha_area:.4f}

Alpha Density: {alpha_density:.2f}

============================================================
"""

        with open(results_file, "w") as f:

            f.write(summary)

        print(summary)

        print(f"💾 Reachable poses saved: {poses_file}")

        if hull_file != "N/A":

            print(f"💾 Alpha boundary saved: {hull_file}")

        print(f"💾 Results summary saved: {results_file}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main(args=None):

    rclpy.init(args=args)

    dof = 5

    if len(sys.argv) > 1:

        dof = int(sys.argv[1])

    analyzer = AdvancedWorkspaceAnalyzer(dof)

    analyzer.run_advanced_analysis(num_samples=10000)

    analyzer.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":

    main()