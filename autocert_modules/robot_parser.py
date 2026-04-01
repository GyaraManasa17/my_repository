import rclpy
import xml.etree.ElementTree as ET
import math
import os
import subprocess

class RobotKinematicParser:
    def __init__(self, node, urdf_path=None, srdf_path=None):
        """
        Takes a ROS 2 Node reference and optional paths to URDF/XACRO and SRDF files.
        """
        self.node = node
        
        # ✅ Default paths (hardcoded for Open Manipulator X), but can be easily overridden!
        self.urdf_path = urdf_path or "/home/ubuntu/omx_ws/src/open_manipulator/open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x.urdf.xacro"
        self.srdf_path = srdf_path or "/home/ubuntu/omx_ws/src/omx_5dof_moveit/config/open_manipulator_x.srdf"

        self.urdf_string = None
        self.srdf_string = None

        self.group_name = None
        self.base_link = None
        self.end_link = None
        self.joints =[]
        self.limits =[]
        self.dof = 0

    def load_and_parse(self,timeout_sec=10.0):
        """
        Reads the URDF/XACRO and SRDF directly from the file system.
        """
        # 1. Load URDF / XACRO
        self.node.get_logger().info(f"📁 Loading URDF from: {self.urdf_path}")
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"❌ URDF/Xacro file not found at: {self.urdf_path}")

        # If it's a xacro file, we must process it using the 'xacro' command line tool
        if self.urdf_path.endswith('.xacro'):
            try:
                self.urdf_string = subprocess.check_output(['xacro', self.urdf_path], text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"❌ Failed to process Xacro file. Is 'xacro' installed? Error: {e}")
        else:
            with open(self.urdf_path, 'r') as f:
                self.urdf_string = f.read()

        # 2. Load SRDF
        self.node.get_logger().info(f"📁 Loading SRDF from: {self.srdf_path}")
        if not os.path.exists(self.srdf_path):
            raise FileNotFoundError(f"❌ SRDF file not found at: {self.srdf_path}")

        with open(self.srdf_path, 'r') as f:
            self.srdf_string = f.read()

        self.node.get_logger().info("✅ URDF and SRDF successfully loaded from files.")

        # 3. Parse them using the exact same logic as before
        self._parse_srdf()
        self._parse_urdf()

    def _parse_srdf(self):
        try:
            srdf_root = ET.fromstring(self.srdf_string)
        except ET.ParseError as e:
            raise RuntimeError(f"❌ SRDF parsing failed (Malformed XML): {e}")

        groups = [g.get("name") for g in srdf_root.findall("group")]

        if "arm" in groups:
            self.group_name = "arm"
        else:
            print("\nAvailable Planning Groups:", groups)
            while True:
                choice = input("✍️ Select planning group: ").strip()
                if choice in groups:
                    self.group_name = choice
                    break
                print("❌ Invalid group. Please type one of the available groups.")

        for g in srdf_root.findall("group"):
            if g.get("name") == self.group_name:
                chain = g.find("chain")
                if chain is not None:
                    self.base_link = chain.get("base_link")
                    self.end_link = chain.get("tip_link")
                else:
                    raise RuntimeError(f"Group '{self.group_name}' must be chain-based.")
        
        if self.base_link is None or self.end_link is None:
            raise RuntimeError(f"❌ Could not determine base_link or tip_link for group '{self.group_name}' from SRDF.")
                    
        print(f"\n🔗 Group: {self.group_name} | Base: {self.base_link} | Tip: {self.end_link}")

    def _parse_urdf(self):
        try:
            urdf_root = ET.fromstring(self.urdf_string)
        except ET.ParseError as e:
            raise RuntimeError(f"❌ URDF parsing failed (Malformed XML): {e}")

        child_parent = {}
        joint_info = {}

        for j in urdf_root.findall("joint"):
            j_type = j.get("type")
            child = j.find("child").get("link")
            parent = j.find("parent").get("link")
            
            child_parent[child] = parent
            
            if j_type != "fixed":
                limit = j.find("limit")
                
                if limit is not None:
                    lower = float(limit.get("lower", -math.pi))
                    upper = float(limit.get("upper", math.pi))
                    joint_info[child] = (j.get("name"), lower, upper)
                elif j_type == "continuous":
                    joint_info[child] = (j.get("name"), -math.pi, math.pi)
                else:
                    self.node.get_logger().warning(f"⚠️ Joint '{j.get('name')}' is missing limits! Defaulting to [-pi, pi].")
                    joint_info[child] = (j.get("name"), -math.pi, math.pi)

        link = self.end_link
        chain =[]
        
        while link != self.base_link:
            if link not in child_parent:
                raise RuntimeError(
                    f"❌ Broken kinematic chain! Cannot trace from '{self.end_link}' to '{self.base_link}'. "
                    f"Traversal stopped at disconnected link: '{link}'"
                )
            if link in joint_info:
                chain.append(joint_info[link])
            link = child_parent[link]

        chain.reverse()

        self.joints = []
        self.limits =[]

        for name, low, up in chain:
            self.joints.append(name)
            self.limits.append((low, up))

        self.dof = len(self.joints)
        
        print(f"\n⚙️ Detected {self.dof}-DOF Manipulator | Active Joints & Limits:")
        for j, l in zip(self.joints, self.limits):
            print(f"  - {j}: [{l[0]:.2f}, {l[1]:.2f}] rad")




# # autocert_modules/robot_parser.py

# import rclpy
# from rclpy.qos import QoSProfile, QoSDurabilityPolicy
# from std_msgs.msg import String
# import xml.etree.ElementTree as ET
# import math
# import time

# class RobotKinematicParser:
#     def __init__(self, node):
#         """
#         Takes a ROS 2 Node reference to create subscriptions.
#         """
#         self.node = node
#         self.urdf_string = None
#         self.srdf_string = None

#         self.group_name = None
#         self.base_link = None
#         self.end_link = None
#         self.joints = []
#         self.limits =[]
#         self.dof = 0  # ✅ IMPROVEMENT 3: Store Degrees of Freedom

#         qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
#         self.node.create_subscription(String, "/robot_description", self._urdf_cb, qos)
#         self.node.create_subscription(String, "/robot_description_semantic", self._srdf_cb, qos)

#     def _urdf_cb(self, msg):
#         self.urdf_string = msg.data

#     def _srdf_cb(self, msg):
#         self.srdf_string = msg.data

#     def wait_and_parse(self, timeout_sec=10.0):
#         # ✅ IMPROVEMENT 1: Added Timeout Protection
#         self.node.get_logger().info(f"⏳ Waiting for URDF and SRDF topics (timeout: {timeout_sec}s)...")
#         start_time = time.time()
        
#         while rclpy.ok() and (self.urdf_string is None or self.srdf_string is None):
#             if time.time() - start_time > timeout_sec:
#                 raise RuntimeError("❌ Timeout waiting for /robot_description and /robot_description_semantic. Is MoveIt running?")
#             rclpy.spin_once(self.node, timeout_sec=0.1)
            
#         self.node.get_logger().info("✅ URDF and SRDF successfully loaded.")

#         self._parse_srdf()
#         self._parse_urdf()

#     def _parse_srdf(self):
#         try:
#             srdf_root = ET.fromstring(self.srdf_string)
#         except ET.ParseError as e:
#             raise RuntimeError(f"❌ SRDF parsing failed (Malformed XML): {e}")

#         groups =[g.get("name") for g in srdf_root.findall("group")]

#         if "arm" in groups:
#             self.group_name = "arm"
#         else:
#             print("\nAvailable Planning Groups:", groups)
#             while True:
#                 choice = input("✍️ Select planning group: ").strip()
#                 if choice in groups:
#                     self.group_name = choice
#                     break
#                 print("❌ Invalid group. Please type one of the available groups.")

#         for g in srdf_root.findall("group"):
#             if g.get("name") == self.group_name:
#                 chain = g.find("chain")
#                 if chain is not None:
#                     self.base_link = chain.get("base_link")
#                     self.end_link = chain.get("tip_link")
#                 else:
#                     raise RuntimeError(f"Group '{self.group_name}' must be chain-based.")
        
#         if self.base_link is None or self.end_link is None:
#             raise RuntimeError(f"❌ Could not determine base_link or tip_link for group '{self.group_name}' from SRDF.")
                    
#         print(f"\n🔗 Group: {self.group_name} | Base: {self.base_link} | Tip: {self.end_link}")

#     def _parse_urdf(self):
#         try:
#             urdf_root = ET.fromstring(self.urdf_string)
#         except ET.ParseError as e:
#             raise RuntimeError(f"❌ URDF parsing failed (Malformed XML): {e}")

#         child_parent = {}
#         joint_info = {}

#         for j in urdf_root.findall("joint"):
#             j_type = j.get("type")
#             child = j.find("child").get("link")
#             parent = j.find("parent").get("link")
            
#             child_parent[child] = parent
            
#             if j_type != "fixed":
#                 limit = j.find("limit")
                
#                 # ✅ IMPROVEMENT 2: Handle Missing Joint Limits Safely
#                 if limit is not None:
#                     # Default to -pi/pi if the tag exists but the attributes 'lower' or 'upper' are missing
#                     lower = float(limit.get("lower", -math.pi))
#                     upper = float(limit.get("upper", math.pi))
#                     joint_info[child] = (j.get("name"), lower, upper)
#                 elif j_type == "continuous":
#                     joint_info[child] = (j.get("name"), -math.pi, math.pi)
#                 else:
#                     self.node.get_logger().warning(f"⚠️ Joint '{j.get('name')}' is missing limits! Defaulting to [-pi, pi].")
#                     joint_info[child] = (j.get("name"), -math.pi, math.pi)

#         link = self.end_link
#         chain =[]
        
#         while link != self.base_link:
#             if link not in child_parent:
#                 raise RuntimeError(
#                     f"❌ Broken kinematic chain! Cannot trace from '{self.end_link}' to '{self.base_link}'. "
#                     f"Traversal stopped at disconnected link: '{link}'"
#                 )
#             if link in joint_info:
#                 chain.append(joint_info[link])
#             link = child_parent[link]

#         chain.reverse()

#         self.joints =[]
#         self.limits =[]

#         for name, low, up in chain:
#             self.joints.append(name)
#             self.limits.append((low, up))

#         # ✅ IMPROVEMENT 3: Store and print the Degree of Freedom (DOF)
#         self.dof = len(self.joints)
        
#         print(f"\n⚙️ Detected {self.dof}-DOF Manipulator | Active Joints & Limits:")
#         for j, l in zip(self.joints, self.limits):
#             print(f"  - {j}: [{l[0]:.2f}, {l[1]:.2f}] rad")




# # autocert_modules/robot_parser.py

# import rclpy
# from rclpy.qos import QoSProfile, QoSDurabilityPolicy
# from std_msgs.msg import String
# import xml.etree.ElementTree as ET

# class RobotKinematicParser:
#     def __init__(self, node):
#         """
#         Takes a ROS 2 Node reference to create subscriptions.
#         """
#         self.node = node
#         self.urdf_string = None
#         self.srdf_string = None

#         # Public properties to be used by other modules
#         self.group_name = None
#         self.base_link = None
#         self.end_link = None
#         self.joints = []
#         self.limits =[]

#         # Subscriptions
#         qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
#         self.node.create_subscription(String, "/robot_description", self._urdf_cb, qos)
#         self.node.create_subscription(String, "/robot_description_semantic", self._srdf_cb, qos)

#     def _urdf_cb(self, msg):
#         self.urdf_string = msg.data

#     def _srdf_cb(self, msg):
#         self.srdf_string = msg.data

#     def wait_and_parse(self):
#         """
#         Blocks until URDF and SRDF are received, then parses them.
#         """
#         self.node.get_logger().info("⏳ Waiting for URDF and SRDF topics...")
#         while rclpy.ok() and (self.urdf_string is None or self.srdf_string is None):
#             rclpy.spin_once(self.node, timeout_sec=0.1)
#         self.node.get_logger().info("✅ URDF and SRDF successfully loaded.")

#         self._parse_srdf()
#         self._parse_urdf()

#     def _parse_srdf(self):
#         srdf_root = ET.fromstring(self.srdf_string)
#         groups =[g.get("name") for g in srdf_root.findall("group")]

#         if "arm" in groups:
#             self.group_name = "arm"
#         else:
#             print("\nAvailable Planning Groups:", groups)
#             self.group_name = input("✍️ Select planning group: ").strip()

#         for g in srdf_root.findall("group"):
#             if g.get("name") == self.group_name:
#                 chain = g.find("chain")
#                 if chain is not None:
#                     self.base_link = chain.get("base_link")
#                     self.end_link = chain.get("tip_link")
#                 else:
#                     raise RuntimeError(f"Group '{self.group_name}' must be chain-based.")
                    
#         print(f"\n🔗 Group: {self.group_name} | Base: {self.base_link} | Tip: {self.end_link}")

#     def _parse_urdf(self):
#         urdf_root = ET.fromstring(self.urdf_string)
#         child_parent, joint_info = {}, {}

#         # 1. Map all joints and limits
#         for j in urdf_root.findall("joint"):
#             limit = j.find("limit")
#             if limit is not None:
#                 child = j.find("child").get("link")
#                 parent = j.find("parent").get("link")
#                 child_parent[child] = parent
#                 joint_info[child] = (j.get("name"), float(limit.get("lower", 0)), float(limit.get("upper", 0)))

#         # 2. Traverse backward from end_link to base_link
#         link, chain = self.end_link,[]
#         while link != self.base_link:
#             if link in joint_info:
#                 chain.append(joint_info[link])
#             link = child_parent.get(link)

#         # 3. Reverse to get Base -> Tip order
#         chain.reverse()
#         for name, low, up in chain:
#             self.joints.append(name)
#             self.limits.append((low, up))

#         print("\n⚙️ Detected Active Joints & Limits:")
#         for j, l in zip(self.joints, self.limits):
#             print(f"  - {j}: [{l[0]:.2f}, {l[1]:.2f}] rad")