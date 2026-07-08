#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from scipy.spatial.transform import Rotation as R

import pandas as pd
import numpy as np
import time
import sys


class ReachabilityVerifier(Node):

    def __init__(self, dof):

        super().__init__("reachability_verifier")

        self.dof = dof

        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")

        self.group_name = "arm"
        self.base_frame = "link1"
        self.end_link = "end_effector_link"

        if dof == 5:
            self.tag = "_5dof"
            self.joints = ["joint1","joint2","joint3","joint4"]

        elif dof == 6:
            self.tag = "_6dof"
            self.joints = ["joint1","joint2","joint3","joint4","joint5_roll"]

        else:
            raise ValueError("DOF must be 5 or 6")


    # --------------------------------------------------

    def wait_for_service(self):

        self.get_logger().info("Waiting for MoveIt /compute_ik service...")

        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            return False

        return True


    # --------------------------------------------------

    def verify_poses(self, csv_5dof, csv_6dof, seed=42):

        if not self.wait_for_service():

            self.get_logger().error("IK service not available!")
            return

        print("\n" + "="*80)
        print(f"IK VERIFICATION USING {self.dof} DOF ROBOT")
        print("="*80)

        np.random.seed(seed)

        print(f"Random Seed: {seed}")

        # --------------------------------------------------
        # LOAD DATASETS
        # --------------------------------------------------

        df5 = pd.read_csv(csv_5dof)
        df5["source"] = "5dof"

        df6 = pd.read_csv(csv_6dof)
        df6["source"] = "6dof"

        print(f"Loaded {len(df5)} poses from 5DOF dataset")
        print(f"Loaded {len(df6)} poses from 6DOF dataset")

        df = pd.concat([df5, df6], ignore_index=True)

        # --------------------------------------------------
        # INITIALIZE JOINT STORAGE
        # --------------------------------------------------

        for j in range(1,5):
            df[f"joint{j}_5"] = None

        # for j in range(1,6):
        #     df[f"joint{j}_6"] = None
        for j in range(1,5):
            df[f"joint{j}_6"] = None

        df["joint5_roll_6"] = None

        # Copy original FK joints

        for j in range(1,5):
            if f"joint{j}" in df5.columns:
                df.loc[df["source"]=="5dof", f"joint{j}_5"] = df.loc[df["source"]=="5dof", f"joint{j}"]

        # for j in range(1,6):
        #     if f"joint{j}" or f"joint{j}_roll" in df6.columns:
        #         if (j==5):
        #             df.loc[df["source"]=="6dof", f"joint{j}_roll_6"] = df.loc[df["source"]=="6dof", f"joint{j}_roll"]
        #         else:
        #             df.loc[df["source"]=="6dof", f"joint{j}_6"] = df.loc[df["source"]=="6dof", f"joint{j}"]

        for j in range(1,6):
            if j == 5:
                if "joint5_roll" in df6.columns:
                    df.loc[df["source"]=="6dof","joint5_roll_6"] = \
                        df.loc[df["source"]=="6dof","joint5_roll"]

            else:
                if f"joint{j}" in df6.columns:
                    df.loc[df["source"]=="6dof",f"joint{j}_6"] = \
                        df.loc[df["source"]=="6dof",f"joint{j}"]

        reach_col = f"reachable_by_{self.dof}dof"

        df[reach_col] = False

        total = len(df)

        print(f"Total poses to test: {total}")

        # --------------------------------------------------
        # IK REQUEST
        # --------------------------------------------------

        req = GetPositionIK.Request()

        ik_req = PositionIKRequest()

        ik_req.group_name = self.group_name
        ik_req.ik_link_name = self.end_link
        ik_req.avoid_collisions = False

        ik_req.timeout.sec = 0
        ik_req.timeout.nanosec = 50000000

        # --------------------------------------------------
        # HOME SEED STATE
        # --------------------------------------------------

        seed_state = JointState()

        seed_state.name = self.joints

        seed_state.position = [0.0]*len(self.joints)

        ik_req.robot_state.joint_state = seed_state

        start = time.time()

        # --------------------------------------------------
        # IK LOOP
        # --------------------------------------------------

        for i,row in df.iterrows():

            if i % 1000 == 0 and i > 0:

                rate = i/(time.time()-start)

                print(f"{i}/{total} poses checked ({rate:.1f} poses/sec)")

            rot = R.from_euler('xyz',[row["roll"],row["pitch"],row["yaw"]])

            qx,qy,qz,qw = rot.as_quat()

            pose = PoseStamped()

            pose.header.frame_id = self.base_frame

            pose.pose.position.x = row["x"]
            pose.pose.position.y = row["y"]
            pose.pose.position.z = row["z"]

            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            ik_req.pose_stamped = pose

            req.ik_request = ik_req

            future = self.ik_client.call_async(req)

            rclpy.spin_until_future_complete(self,future)

            resp = future.result()

            if resp and resp.error_code.val == 1:

                df.at[i,reach_col] = True

                names = resp.solution.joint_state.name
                values = resp.solution.joint_state.position

                joint_map = dict(zip(names,values))

                if self.dof == 5:

                    for j in range(1,5):
                        df.at[i,f"joint{j}_5"] = joint_map.get(f"joint{j}")

                if self.dof == 6:

                    for j in range(1,6):
                        if (j==5):
                            df.at[i,f"joint{j}_roll_6"] = joint_map.get(f"joint{j}_roll")
                        else:
                            df.at[i,f"joint{j}_6"] = joint_map.get(f"joint{j}")

        elapsed = time.time()-start

        # --------------------------------------------------
        # STATISTICS
        # --------------------------------------------------

        print("\n"+"="*80)
        print("RESULTS")
        print("="*80)

        print(f"Robot used: {self.dof}DOF")
        print(f"Time: {elapsed:.2f} sec\n")

        df5_sub = df[df["source"]=="5dof"]

        s5 = df5_sub[reach_col].sum()

        print(f"5DOF poses: {len(df5_sub)}")
        print(f"Reachable: {s5} ({(s5/len(df5_sub))*100:.1f}%)")

        df6_sub = df[df["source"]=="6dof"]

        s6 = df6_sub[reach_col].sum()

        print(f"\n6DOF poses: {len(df6_sub)}")
        print(f"Reachable: {s6} ({(s6/len(df6_sub))*100:.1f}%)")
        print(f"Unreachable: {len(df6_sub)-s6}")

        print("="*80)

        # --------------------------------------------------
        # SAVE DATASET
        # --------------------------------------------------
        df = df.drop(columns=["joint1","joint2","joint3","joint4","joint5_roll"])
        
        out = f"verified_kinematic_overlap{self.tag}.csv"

        df.to_csv(out,index=False)

        print(f"Saved dataset → {out}")


# --------------------------------------------------

def main(args=None):

    rclpy.init(args=args)

    dof = 5

    if len(sys.argv) > 1:
        dof = int(sys.argv[1])

    verifier = ReachabilityVerifier(dof)

    csv_5dof = "omx_all_reachable_poses_5dof.csv"
    csv_6dof = "omx_all_reachable_poses_6dof.csv"

    verifier.verify_poses(csv_5dof,csv_6dof)

    verifier.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()