import rclpy
import xml.etree.ElementTree as ET
import math
import os
import subprocess

class RobotKinematicParser:
    def __init__(self, node, urdf_path=None, srdf_path=None, group_name=None):
        """
        Takes a ROS 2 Node reference, optional paths to URDF/XACRO and SRDF files, 
        and an optional planning group name.
        """
        self.node = node
        
        # ✅ Default paths (hardcoded for Open Manipulator X), but can be easily overridden!
        self.urdf_path = urdf_path or "/home/ubuntu/omx_ws/src/open_manipulator/open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x.urdf.xacro"
        self.srdf_path = srdf_path or "/home/ubuntu/omx_ws/src/omx_5dof_moveit/config/open_manipulator_x.srdf"

        self.urdf_string = None
        self.srdf_string = None

        self.group_name = group_name  # Passed as an argument to avoid blocking input
        self.base_link = None
        self.end_link = None
        self.joints = []
        self.limits =[]
        self.dof = 0

    def load_and_parse(self):
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

        # 3. Parse them using the exact same logic
        self._parse_srdf()
        self._parse_urdf()

    def _parse_srdf(self):
        try:
            srdf_root = ET.fromstring(self.srdf_string)
        except ET.ParseError as e:
            raise RuntimeError(f"❌ SRDF parsing failed (Malformed XML): {e}")

        groups =[g.get("name") for g in srdf_root.findall("group")]

        # Smart Group Selection (No blocking input!)
        if self.group_name:
            if self.group_name not in groups:
                raise ValueError(f"❌ Provided group_name '{self.group_name}' not found. Available: {groups}")
        else:
            if "arm" in groups:
                self.group_name = "arm"
            elif groups:
                self.group_name = groups[0]
                self.node.get_logger().warning(f"⚠️ 'arm' not found. Defaulting to first available group: '{self.group_name}'")
            else:
                raise ValueError("❌ No planning groups found in SRDF.")

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
                    
        self.node.get_logger().info(f"🔗 Group: {self.group_name} | Base: {self.base_link} | Tip: {self.end_link}")

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
                    self.node.get_logger().warning(f"⚠️ Joint '{j.get('name')}' is missing limits! Defaulting to[-pi, pi].")
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
        
        self.node.get_logger().info(f"⚙️ Detected {self.dof}-DOF Manipulator | Active Joints & Limits:")
        for j, l in zip(self.joints, self.limits):
            self.node.get_logger().info(f"  - {j}: [{l[0]:.2f}, {l[1]:.2f}] rad")