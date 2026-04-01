#include <rclcpp/rclcpp.hpp>
#include <moveit/warehouse/planning_scene_storage.hpp>
#include <warehouse_ros/database_loader.h>

#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp> // Replaced Cartesian with Joint Constraints

#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Modify these if your URDF/SRDF has different names for the two arms
const std::string ROBOT_NAME_5 = "open_manipulator_x";
const std::string PLANNING_GROUP_5 = "arm";
const std::string SCENE_NAME_5 = "benchmark_scene_5dof";

const std::string ROBOT_NAME_6 = "open_manipulator_x"; // Update if your 6dof arm has a different robot_model_name
const std::string PLANNING_GROUP_6 = "arm";
const std::string SCENE_NAME_6 = "benchmark_scene_6dof";

// Helper function to create and upload a Joint Space Motion Planning Request
void addJointSpaceQuery(moveit_warehouse::PlanningSceneStorage& ps_storage,
                        const std::string& scene_name,
                        const std::string& group_name,
                        const std::vector<std::string>& joint_names,
                        const std::vector<double>& goal_positions,
                        const std::string& query_name)
{
    moveit_msgs::msg::MotionPlanRequest req;

    req.group_name = group_name;
    req.num_planning_attempts = 1;
    req.allowed_planning_time = 5.0;

    // ==============================
    // START STATE (HOME POSITION)
    // ==============================
    req.start_state.is_diff = false;
    req.start_state.joint_state.name = joint_names;
    // Fill with exactly 0.0 for all joints
    req.start_state.joint_state.position.assign(joint_names.size(), 0.0);

    // ==============================
    // WORKSPACE BOUNDS
    // ==============================
    req.workspace_parameters.header.frame_id = "link1";
    req.workspace_parameters.min_corner.x = -1.0;
    req.workspace_parameters.min_corner.y = -1.0;
    req.workspace_parameters.min_corner.z = -1.0;

    req.workspace_parameters.max_corner.x = 1.0;
    req.workspace_parameters.max_corner.y = 1.0;
    req.workspace_parameters.max_corner.z = 1.0;

    // ==============================
    // GOAL JOINT CONSTRAINTS
    // ==============================
    moveit_msgs::msg::Constraints goal;

    for (size_t i = 0; i < joint_names.size(); ++i)
    {
        moveit_msgs::msg::JointConstraint jc;
        jc.joint_name = joint_names[i];
        jc.position = goal_positions[i];
        jc.tolerance_above = 0.01;
        jc.tolerance_below = 0.01;
        jc.weight = 1.0;
        
        goal.joint_constraints.push_back(jc);
    }

    req.goal_constraints.push_back(goal);

    // Upload to Database
    ps_storage.addPlanningQuery(req, scene_name, query_name);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("moveit_warehouse_uploader");

    node->declare_parameter("warehouse_plugin", "warehouse_ros_mongo::MongoDatabaseConnection");
    node->declare_parameter("warehouse_host", "moveit_mongo");
    node->declare_parameter("warehouse_port", 27017);

    warehouse_ros::DatabaseLoader db_loader(node);
    auto db_conn = db_loader.loadDatabase();

    if (!db_conn)
    {
        RCLCPP_ERROR(node->get_logger(), "Failed to load DB plugin");
        return 1;
    }

    db_conn->setParams("moveit_mongo", 27017, 5.0);

    if (!db_conn->connect())
    {
        RCLCPP_ERROR(node->get_logger(), "Failed to connect to MongoDB");
        return 1;
    }

    RCLCPP_INFO(node->get_logger(), "Connected to MongoDB");

    moveit_warehouse::PlanningSceneStorage ps_storage(db_conn);

    // ==============================
    // INITIALIZE 5-DOF SCENE
    // ==============================
    try {
        ps_storage.removePlanningScene(SCENE_NAME_5);
        ps_storage.removePlanningQueries(SCENE_NAME_5);
    } catch (...) {}

    moveit_msgs::msg::PlanningScene scene5;
    scene5.name = SCENE_NAME_5;
    scene5.robot_model_name = ROBOT_NAME_5;
    scene5.is_diff = true;
    ps_storage.addPlanningScene(scene5);

    // ==============================
    // INITIALIZE 6-DOF SCENE
    // ==============================
    try {
        ps_storage.removePlanningScene(SCENE_NAME_6);
        ps_storage.removePlanningQueries(SCENE_NAME_6);
    } catch (...) {}

    moveit_msgs::msg::PlanningScene scene6;
    scene6.name = SCENE_NAME_6;
    scene6.robot_model_name = ROBOT_NAME_6;
    scene6.is_diff = true;
    ps_storage.addPlanningScene(scene6);

    RCLCPP_INFO(node->get_logger(), "Separate scenes created for 5-DOF and 6-DOF benchmarks");

    std::ifstream file("/home/ubuntu/metric_ws/poses_storage/shared_300_poses_for_both_arms.csv");

    if (!file.is_open())
    {
        RCLCPP_ERROR(node->get_logger(), "CSV not found");
        return 1;
    }

    std::string line;
    std::getline(file, line); // skip header

    int total_rows = 0;
    int count_5dof = 0;
    int count_6dof = 0;

    // Define standard MoveIt joint names expected by your SRDFs
    std::vector<std::string> joint_names_5 = {"joint1", "joint2", "joint3", "joint4"};
    std::vector<std::string> joint_names_6 = {"joint1", "joint2", "joint3", "joint4", "joint5_roll"};

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> row;

        while (std::getline(ss, token, ','))
            row.push_back(token);

        // Ensure we have all columns up to reachable_by_6dof (index 16)
        if (row.size() < 16) continue;

        std::string query_name = "pose_" + std::to_string(total_rows + 1);

        // ------------------------------------------
        // Extract 5-DOF Joints (Columns 7 to 10)
        // ------------------------------------------
        try
        {
            // Skip parsing if the cell says 'nan' or is empty
            if (!row[7].empty() && row[7] != "nan" && row[7] != "NaN")
            {
                std::vector<double> goal_5 = {
                    std::stod(row[7]), std::stod(row[8]), 
                    std::stod(row[9]), std::stod(row[10])
                };

                addJointSpaceQuery(ps_storage, SCENE_NAME_5, PLANNING_GROUP_5, joint_names_5, goal_5, query_name);
                count_5dof++;
            }
        }
        catch (...) {}

        // ------------------------------------------
        // Extract 6-DOF Joints (Columns 11 to 15)
        // ------------------------------------------
        try
        {
            if (!row[11].empty() && row[11] != "nan" && row[11] != "NaN")
            {
                std::vector<double> goal_6 = {
                    std::stod(row[11]), std::stod(row[12]), 
                    std::stod(row[13]), std::stod(row[14]), std::stod(row[15])
                };

                addJointSpaceQuery(ps_storage, SCENE_NAME_6, PLANNING_GROUP_6, joint_names_6, goal_6, query_name);
                count_6dof++;
            }
        }
        catch (...) {}

        total_rows++;
    }

    RCLCPP_INFO(node->get_logger(), "Processed %d total CSV rows.", total_rows);
    RCLCPP_INFO(node->get_logger(), "Uploaded %d queries to %s.", count_5dof, SCENE_NAME_5.c_str());
    RCLCPP_INFO(node->get_logger(), "Uploaded %d queries to %s.", count_6dof, SCENE_NAME_6.c_str());

    rclcpp::shutdown();
    return 0;
}





// #include <rclcpp/rclcpp.hpp>

// #include <moveit/warehouse/planning_scene_storage.hpp>
// #include <warehouse_ros/database_loader.h>

// #include <moveit_msgs/msg/motion_plan_request.hpp>
// #include <moveit_msgs/msg/constraints.hpp>
// #include <moveit_msgs/msg/joint_constraint.hpp>

// #include <fstream>
// #include <sstream>
// #include <vector>

// const std::string PLANNING_GROUP = "arm";

// struct PoseRow
// {
//   double j1_5,j2_5,j3_5,j4_5;
//   double j1_6,j2_6,j3_6,j4_6,j5_6;
// };

// std::vector<PoseRow> loadCSV(const std::string &file)
// {
//   std::vector<PoseRow> data;

//   std::ifstream in(file);
//   std::string line;

//   std::getline(in,line);

//   while(std::getline(in,line))
//   {
//     std::stringstream ss(line);
//     std::string token;
//     std::vector<std::string> row;

//     while(std::getline(ss,token,','))
//       row.push_back(token);

//     PoseRow p;

//     p.j1_5 = std::stod(row[7]);
//     p.j2_5 = std::stod(row[8]);
//     p.j3_5 = std::stod(row[9]);
//     p.j4_5 = std::stod(row[10]);

//     p.j1_6 = std::stod(row[11]);
//     p.j2_6 = std::stod(row[12]);
//     p.j3_6 = std::stod(row[13]);
//     p.j4_6 = std::stod(row[14]);
//     p.j5_6 = std::stod(row[15]);

//     data.push_back(p);
//   }

//   return data;
// }

// moveit_msgs::msg::Constraints buildJointGoal(
//         const std::vector<std::string>& joints,
//         const std::vector<double>& values)
// {
//   moveit_msgs::msg::Constraints goal;

//   for(size_t i=0;i<joints.size();i++)
//   {
//     moveit_msgs::msg::JointConstraint jc;

//     jc.joint_name = joints[i];
//     jc.position = values[i];
//     jc.tolerance_above = 0.001;
//     jc.tolerance_below = 0.001;
//     jc.weight = 1.0;

//     goal.joint_constraints.push_back(jc);
//   }

//   return goal;
// }

// int main(int argc,char** argv)
// {
//   rclcpp::init(argc,argv);

//   auto node = std::make_shared<rclcpp::Node>("warehouse_query_uploader");

//   node->declare_parameter("warehouse_plugin",
//         "warehouse_ros_mongo::MongoDatabaseConnection");

//   node->declare_parameter("warehouse_host","localhost");
//   node->declare_parameter("warehouse_port",33829);

//   warehouse_ros::DatabaseLoader db_loader(node);
//   auto db_conn = db_loader.loadDatabase();

//   if(!db_conn)
//   {
//     RCLCPP_ERROR(node->get_logger(),"Failed to load warehouse plugin");
//     return 1;
//   }

//   if(!db_conn->connect())
//   {
//     RCLCPP_ERROR(node->get_logger(),"Failed to connect to database");
//     return 1;
//   }

//   moveit_warehouse::PlanningSceneStorage ps_storage(db_conn);

//   const std::string scene5 = "scene_5dof";
//   const std::string scene6 = "scene_6dof";

//   auto rows = loadCSV(
//     "/home/ubuntu/metric_ws/poses_storage/shared_300_poses_for_both_arms.csv");

//   RCLCPP_INFO(node->get_logger(),"Loaded %ld poses",rows.size());

//   const std::vector<std::string> joints5 =
//   {"joint1","joint2","joint3","joint4"};

//   const std::vector<std::string> joints6 =
//   {"joint1","joint2","joint3","joint4","joint5_roll"};

//   int id = 0;

//   for(const auto &row : rows)
//   {
//     std::vector<double> j5 =
//     {row.j1_5,row.j2_5,row.j3_5,row.j4_5};

//     std::vector<double> j6 =
//     {row.j1_6,row.j2_6,row.j3_6,row.j4_6,row.j5_6};

//     auto goal5 = buildJointGoal(joints5,j5);
//     auto goal6 = buildJointGoal(joints6,j6);

//     moveit_msgs::msg::MotionPlanRequest req5;
//     moveit_msgs::msg::MotionPlanRequest req6;

//     req5.group_name = PLANNING_GROUP;
//     req6.group_name = PLANNING_GROUP;

//     req5.num_planning_attempts = 1;
//     req6.num_planning_attempts = 1;

//     req5.allowed_planning_time = 5.0;
//     req6.allowed_planning_time = 5.0;

//     req5.goal_constraints.push_back(goal5);
//     req6.goal_constraints.push_back(goal6);

//     req5.start_state.is_diff = false;
//     req6.start_state.is_diff = false;

//     req5.start_state.joint_state.name =
//     {"joint1","joint2","joint3","joint4","joint5_roll"};

//     req6.start_state.joint_state.name =
//     {"joint1","joint2","joint3","joint4","joint5_roll"};

//     req5.start_state.joint_state.position =
//     {0,0,0,0,0};

//     req6.start_state.joint_state.position =
//     {0,0,0,0,0};

//     std::string q5 = "query5_" + std::to_string(id);
//     std::string q6 = "query6_" + std::to_string(id);

//     ps_storage.addPlanningQuery(req5,scene5,q5);
//     ps_storage.addPlanningQuery(req6,scene6,q6);

//     id++;
//   }

//   RCLCPP_INFO(node->get_logger(),
//               "Stored %d planning queries for each robot",id);

//   rclcpp::shutdown();
// }




// #include <rclcpp/rclcpp.hpp>

// #include <moveit/planning_scene/planning_scene.h>
// #include <moveit/planning_scene_monitor/planning_scene_monitor.h>
// #include <moveit/robot_state/robot_state.h>

// #include <moveit_msgs/msg/constraints.hpp>
// #include <moveit_msgs/msg/joint_constraint.hpp>

// #include <moveit/warehouse/planning_scene_storage.h>
// #include <moveit/warehouse/planning_scene_world_storage.h>
// #include <moveit/warehouse/constraints_storage.h>

// // #include <warehouse_ros_mongo/mongo_database_connection.h>

// #include <fstream>
// #include <sstream>
// #include <vector>

// struct PoseRow
// {
//   double j1_5,j2_5,j3_5,j4_5;
//   double j1_6,j2_6,j3_6,j4_6,j5_6;
// };

// std::vector<PoseRow> loadCSV(const std::string &file)
// {
//   std::vector<PoseRow> data;

//   std::ifstream in(file);
//   std::string line;

//   std::getline(in,line); // skip header

//   while(std::getline(in,line))
//   {
//     std::stringstream ss(line);
//     std::string token;
//     std::vector<std::string> row;

//     while(std::getline(ss,token,','))
//       row.push_back(token);

//     PoseRow p;

//     p.j1_5 = std::stod(row[7]);
//     p.j2_5 = std::stod(row[8]);
//     p.j3_5 = std::stod(row[9]);
//     p.j4_5 = std::stod(row[10]);

//     p.j1_6 = std::stod(row[11]);
//     p.j2_6 = std::stod(row[12]);
//     p.j3_6 = std::stod(row[13]);
//     p.j4_6 = std::stod(row[14]);
//     p.j5_6 = std::stod(row[15]);

//     data.push_back(p);
//   }

//   return data;
// }

// moveit_msgs::msg::Constraints buildConstraint(
//     const std::vector<std::string>& joints,
//     const std::vector<double>& values)
// {
//   moveit_msgs::msg::Constraints c;

//   for(size_t i=0;i<joints.size();i++)
//   {
//     moveit_msgs::msg::JointConstraint jc;

//     jc.joint_name = joints[i];
//     jc.position = values[i];
//     jc.tolerance_above = 0.001;
//     jc.tolerance_below = 0.001;
//     jc.weight = 1.0;

//     c.joint_constraints.push_back(jc);
//   }

//   return c;
// }

// int main(int argc,char** argv)
// {
//   rclcpp::init(argc,argv);
//   auto node = rclcpp::Node::make_shared("store_joint_queries");

//   std::string csv_file = "/home/ubuntu/metric_ws/poses_storage/shared_300_poses_for_both_arms.csv";

//   auto rows = loadCSV(csv_file);

//   RCLCPP_INFO(node->get_logger(),"Loaded %ld poses",rows.size());

// //   warehouse_ros_mongo::MongoDatabaseConnection conn;

//   warehouse_ros::DatabaseLoader db_loader(node);
//   auto db_conn = db_loader.loadDatabase();
//   conn.setParams("localhost",33829);
//   conn.connect();

//   moveit_warehouse::PlanningSceneStorage scene_storage(conn);
//   moveit_warehouse::ConstraintsStorage constraints_storage(conn);

//   const std::vector<std::string> joints5 =
//   {
//     "joint1","joint2","joint3","joint4"
//   };

//   const std::vector<std::string> joints6 =
//   {
//     "joint1","joint2","joint3","joint4","joint5_roll"
//   };

//   int id = 0;

//   for(const auto &row : rows)
//   {
//     std::vector<double> j5 =
//     {
//       row.j1_5,row.j2_5,row.j3_5,row.j4_5
//     };

//     std::vector<double> j6 =
//     {
//       row.j1_6,row.j2_6,row.j3_6,row.j4_6,row.j5_6
//     };

//     auto c5 = buildConstraint(joints5,j5);
//     auto c6 = buildConstraint(joints6,j6);

//     std::string name5 = "query5_" + std::to_string(id);
//     std::string name6 = "query6_" + std::to_string(id);

//     constraints_storage.addConstraints(c5,"scene_5dof",name5);
//     constraints_storage.addConstraints(c6,"scene_6dof",name6);

//     id++;
//   }

//   RCLCPP_INFO(node->get_logger(),"Stored %d queries for each robot",id);

//   rclcpp::shutdown();
//   return 0;
// }