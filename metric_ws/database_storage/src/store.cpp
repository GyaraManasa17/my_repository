#include <rclcpp/rclcpp.hpp>
#include <moveit/warehouse/planning_scene_storage.hpp>
#include <warehouse_ros/database_loader.h>

#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/position_constraint.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <tf2/LinearMath/Quaternion.h>

#include <fstream>
#include <sstream>
#include <vector>

const std::string ROBOT_NAME = "open_manipulator_x";
const std::string PLANNING_GROUP = "arm";
const std::string EE_LINK = "end_effector_link";
const std::string SCENE_NAME = "benchmark_scene";

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

    // Remove old scene
    try
    {
        ps_storage.removePlanningScene(SCENE_NAME);
        ps_storage.removePlanningQueries(SCENE_NAME);
    }
    catch (...)
    {
    }

    moveit_msgs::msg::PlanningScene scene;
    scene.name = SCENE_NAME;
    scene.robot_model_name = ROBOT_NAME;
    scene.is_diff = false;

    ps_storage.addPlanningScene(scene);

    RCLCPP_INFO(node->get_logger(), "Scene created");

    std::ifstream file("/home/ubuntu/metric_ws/workspace/shared_300_poses_for_both_arms.csv");

    if (!file.is_open())
    {
        RCLCPP_ERROR(node->get_logger(), "CSV not found");
        return 1;
    }

    std::string line;
    std::getline(file, line); // skip header

    int count = 0;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> row;

        while (std::getline(ss, token, ','))
            row.push_back(token);

        if (row.size() < 6)
            continue;

        try
        {
            double x = std::stod(row[0]);
            double y = std::stod(row[1]);
            double z = std::stod(row[2]);
            double roll = std::stod(row[3]);
            double pitch = std::stod(row[4]);
            double yaw = std::stod(row[5]);

            tf2::Quaternion q;
            q.setRPY(roll, pitch, yaw);

            moveit_msgs::msg::MotionPlanRequest req;

            req.group_name = PLANNING_GROUP;
            req.num_planning_attempts = 1;
            req.allowed_planning_time = 5.0;

            // ==============================
            // START STATE (HOME POSITION)
            // ==============================

            req.start_state.is_diff = false;

            req.start_state.joint_state.name = {
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5_roll"
            };

            req.start_state.joint_state.position = {
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            };

            // ==============================
            // WORKSPACE BOUNDS
            // ==============================

            req.workspace_parameters.header.frame_id = "link1";

            req.workspace_parameters.min_corner.x = -1;
            req.workspace_parameters.min_corner.y = -1;
            req.workspace_parameters.min_corner.z = -1;

            req.workspace_parameters.max_corner.x = 1;
            req.workspace_parameters.max_corner.y = 1;
            req.workspace_parameters.max_corner.z = 1;

            // ==============================
            // POSITION CONSTRAINT
            // ==============================

            moveit_msgs::msg::Constraints goal;

            moveit_msgs::msg::PositionConstraint pcm;

            pcm.header.frame_id = "link1";
            pcm.link_name = EE_LINK;
            pcm.weight = 1.0;

            pcm.constraint_region.primitives.resize(1);
            pcm.constraint_region.primitives[0].type =
                shape_msgs::msg::SolidPrimitive::SPHERE;

            pcm.constraint_region.primitives[0].dimensions.resize(1);
            pcm.constraint_region.primitives[0].dimensions[0] = 0.001;

            pcm.constraint_region.primitive_poses.resize(1);

            pcm.constraint_region.primitive_poses[0].position.x = x;
            pcm.constraint_region.primitive_poses[0].position.y = y;
            pcm.constraint_region.primitive_poses[0].position.z = z;

            goal.position_constraints.push_back(pcm);

            // ==============================
            // ORIENTATION CONSTRAINT
            // ==============================

            moveit_msgs::msg::OrientationConstraint ocm;

            ocm.header.frame_id = "link1";
            ocm.link_name = EE_LINK;

            ocm.orientation.x = q.x();
            ocm.orientation.y = q.y();
            ocm.orientation.z = q.z();
            ocm.orientation.w = q.w();

            ocm.absolute_x_axis_tolerance = 0.01;
            ocm.absolute_y_axis_tolerance = 0.01;
            ocm.absolute_z_axis_tolerance = 0.01;

            ocm.weight = 1.0;

            goal.orientation_constraints.push_back(ocm);

            req.goal_constraints.push_back(goal);

            std::string query_name = "pose_" + std::to_string(count + 1);

            ps_storage.addPlanningQuery(req, SCENE_NAME, query_name);

            count++;
        }
        catch (...)
        {
            continue;
        }
    }

    RCLCPP_INFO(node->get_logger(),
                "Uploaded %d planning queries",
                count);

    rclcpp::shutdown();
}





// #include <rclcpp/rclcpp.hpp>
// #include <moveit/warehouse/planning_scene_storage.hpp> // Fixed header
// #include <warehouse_ros/database_loader.h>
// #include <moveit_msgs/msg/planning_scene.hpp>
// #include <moveit_msgs/msg/motion_plan_request.hpp>
// #include <geometry_msgs/msg/pose.hpp>
// #include <shape_msgs/msg/solid_primitive.hpp>
// #include <tf2/LinearMath/Quaternion.h>
// #include <fstream>
// #include <sstream>
// #include <string>
// #include <vector>

// // =================================================================
// // ⚠️ UPDATE THESE CONSTANTS TO MATCH YOUR ROBOT SRDF/URDF ⚠️
// // =================================================================
// const std::string ROBOT_NAME = "open_manipulator_x";          // e.g., "panda", "ur5e"
// const std::string PLANNING_GROUP = "arm";   // e.g., "panda_arm", "ur_manipulator"
// const std::string EE_LINK = "end_effector_link";                // e.g., "panda_link8", "tool0"
// const std::string SCENE_NAME = "benchmark_scene";   // Name of the scene in RViz
// // =================================================================

// int main(int argc, char **argv)
// {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<rclcpp::Node>("moveit_warehouse_uploader");

//     // 1. Initialize Warehouse Database Connection Plugin
//     node->declare_parameter("warehouse_plugin", "warehouse_ros_mongo::MongoDatabaseConnection");
//     node->declare_parameter("warehouse_host", "moveit_mongo");
//     node->declare_parameter("warehouse_port", 27017);

//     warehouse_ros::DatabaseLoader db_loader(node);
//     auto db_conn = db_loader.loadDatabase();
    
//     if (!db_conn) {
//         RCLCPP_ERROR(node->get_logger(), "❌ Failed to load Warehouse ROS plugin!");
//         return 1;
//     }

//     db_conn->setParams("moveit_mongo", 27017, 5.0);
//     if (!db_conn->connect()) {
//         RCLCPP_ERROR(node->get_logger(), "❌ Failed to connect to MongoDB at moveit_mongo:27017");
//         return 1;
//     }
//     RCLCPP_INFO(node->get_logger(), "🟢 Connected to MoveIt Warehouse MongoDB.");

//     // 2. Setup Planning Scene Storage
//     // Fixed namespace from moveit_ros_warehouse to moveit_warehouse
//     moveit_warehouse::PlanningSceneStorage ps_storage(db_conn);

//     // 3. Create a blank Planning Scene to attach the queries to
//     moveit_msgs::msg::PlanningScene scene;
//     scene.name = SCENE_NAME;
//     scene.robot_model_name = ROBOT_NAME;
//     scene.is_diff = true;
    
//     // Remove old scene/queries if they exist to avoid clutter
//     try {
//         ps_storage.removePlanningScene(SCENE_NAME);
//         ps_storage.removePlanningQueries(SCENE_NAME);
//     } catch (...) { /* Ignore if it doesn't exist yet */ }
    
//     ps_storage.addPlanningScene(scene);
//     RCLCPP_INFO(node->get_logger(), "Created Planning Scene: '%s'", SCENE_NAME.c_str());

//     // 4. Parse CSV and Upload Queries
//     std::string csv_path = "/home/ubuntu/metric_ws/workspace/shared_300_poses_for_both_arms.csv";
//     std::ifstream file(csv_path);
    
//     if (!file.is_open()) {
//         RCLCPP_ERROR(node->get_logger(), "❌ Failed to open CSV file: %s", csv_path.c_str());
//         return 1;
//     }

//     std::string line;
//     std::getline(file, line); // Skip header row

//     int count = 0;
//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         std::string token;
//         std::vector<std::string> row;
//         while (std::getline(ss, token, ',')) row.push_back(token);

//         if (row.size() >= 7) {
//             try {
//                 double x = std::stod(row[0]);
//                 double y = std::stod(row[1]);
//                 double z = std::stod(row[2]);
//                 double roll = std::stod(row[3]);
//                 double pitch = std::stod(row[4]);
//                 double yaw = std::stod(row[5]);
                
//                 tf2::Quaternion q;
//                 q.setRPY(roll, pitch, yaw);

//                 // Create the standard MoveIt Motion Plan Request (Query)
//                 moveit_msgs::msg::MotionPlanRequest req;
//                 req.group_name = PLANNING_GROUP;
//                 req.num_planning_attempts = 1;
//                 req.allowed_planning_time = 5.0;

//                 req.start_state.is_diff = true;

//                 moveit_msgs::msg::Constraints goal;

//                 // Position Constraints
//                 moveit_msgs::msg::PositionConstraint pcm;
//                 pcm.header.frame_id = "link1";
//                 pcm.link_name = EE_LINK;
//                 pcm.weight = 1.0;
//                 pcm.constraint_region.primitives.resize(1);
//                 pcm.constraint_region.primitives[0].type = shape_msgs::msg::SolidPrimitive::SPHERE;
//                 pcm.constraint_region.primitives[0].dimensions = {0.001}; // 1mm radius tolerance
//                 pcm.constraint_region.primitive_poses.resize(1);
//                 pcm.constraint_region.primitive_poses[0].position.x = x;
//                 pcm.constraint_region.primitive_poses[0].position.y = y;
//                 pcm.constraint_region.primitive_poses[0].position.z = z;
//                 goal.position_constraints.push_back(pcm);

//                 // Orientation Constraints
//                 moveit_msgs::msg::OrientationConstraint ocm;
//                 ocm.header.frame_id = "link1";
//                 ocm.link_name = EE_LINK;
//                 ocm.weight = 1.0;
//                 ocm.orientation.x = q.x();
//                 ocm.orientation.y = q.y();
//                 ocm.orientation.z = q.z();
//                 ocm.orientation.w = q.w();
//                 ocm.absolute_x_axis_tolerance = 0.01; // tolerance in radians
//                 ocm.absolute_y_axis_tolerance = 0.01;
//                 ocm.absolute_z_axis_tolerance = 0.01;
//                 goal.orientation_constraints.push_back(ocm);

//                 req.goal_constraints.push_back(goal);

//                 // Add Query to the Scene in the Warehouse
//                 std::string query_name = "pose_" + std::to_string(count + 1);
//                 ps_storage.addPlanningQuery(req, SCENE_NAME, query_name);
                
//                 count++;
//             } catch (...) {
//                 continue; // Skip malformed rows
//             }
//         }
//     }
//     RCLCPP_INFO(node->get_logger(), "✅ Success! Uploaded %d Planning Queries to Scene '%s'", count, SCENE_NAME.c_str());

//     rclcpp::shutdown();
//     return 0;
// }




// #include <rclcpp/rclcpp.hpp>
// #include <mongocxx/client.hpp>
// #include <mongocxx/instance.hpp>
// #include <mongocxx/uri.hpp>
// #include <bsoncxx/builder/stream/document.hpp>
// #include <tf2/LinearMath/Quaternion.h>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// #include <fstream>
// #include <sstream>
// #include <string>
// #include <vector>
// #include <stdexcept> // Needed for std::exception

// using bsoncxx::builder::stream::document;
// using bsoncxx::builder::stream::finalize;

// class MongoPoseUploader : public rclcpp::Node
// {
// public:
//     MongoPoseUploader() 
//     // UPDATED: Use the Docker container hostname 'moveit_mongo' instead of 'localhost'
//     : Node("mongo_pose_uploader"),
//       client_(mongocxx::uri("mongodb://moveit_mongo:27017"))
//     {
//         db_ = client_["benchmark_db"];
//         coll_ = db_["shared_300_poses"];
        
//         coll_.delete_many({});
//         RCLCPP_INFO(this->get_logger(), "🟢 Connected to MongoDB. Cleared old collection.");

//         std::string csv_path = "/home/ubuntu/metric_ws/workspace/shared_300_poses_for_both_arms.csv";
//         std::ifstream file(csv_path);
        
//         if (!file.is_open()) {
//             RCLCPP_ERROR(this->get_logger(), "❌ Failed to open CSV file: %s", csv_path.c_str());
//             return;
//         }

//         std::string line;
//         std::getline(file, line); // Skip the header row

//         int count = 0;
//         std::vector<bsoncxx::document::value> docs;

//         while (std::getline(file, line)) {
//             std::stringstream ss(line);
//             std::string token;
//             std::vector<std::string> row;
            
//             while (std::getline(ss, token, ',')) {
//                 row.push_back(token);
//             }

//             if (row.size() >= 7) {
//                 try {
//                     // Safe conversion: will throw and be caught if data is invalid (e.g., NaN, empty)
//                     double x = std::stod(row[0]);
//                     double y = std::stod(row[1]);
//                     double z = std::stod(row[2]);
//                     double roll = std::stod(row[3]);
//                     double pitch = std::stod(row[4]);
//                     double yaw = std::stod(row[5]);
//                     std::string source = row[6];

//                     tf2::Quaternion q;
//                     q.setRPY(roll, pitch, yaw); 

//                     auto builder = document{};
//                     bsoncxx::document::value doc_value = builder
//                         << "pose_id" << count + 1
//                         << "x" << x
//                         << "y" << y
//                         << "z" << z
//                         << "qx" << q.x()
//                         << "qy" << q.y()
//                         << "qz" << q.z()
//                         << "qw" << q.w()
//                         << "source" << source
//                         << finalize;

//                     docs.push_back(doc_value);
//                     count++;
//                 } 
//                 catch (const std::exception& e) {
//                     // Catches std::invalid_argument and std::out_of_range
//                     RCLCPP_WARN(this->get_logger(), "⚠️ Skipping malformed CSV row. Error: %s", e.what());
//                     continue; // Skip this row and proceed to the next one
//                 }
//             }
//         }

//         if (!docs.empty()) {
//             coll_.insert_many(docs);
//             RCLCPP_INFO(this->get_logger(), "✅ Success! Batch uploaded %d poses to MongoDB[benchmark_db -> shared_300_poses]", count);
//         } else {
//             RCLCPP_WARN(this->get_logger(), "⚠️ No valid poses found in the CSV to upload.");
//         }
//     }

// private:
//     mongocxx::client client_;
//     mongocxx::database db_;
//     mongocxx::collection coll_;
// };

// int main(int argc, char **argv)
// {
//     rclcpp::init(argc, argv);
//     mongocxx::instance inst{};
//     auto node = std::make_shared<MongoPoseUploader>();
//     rclcpp::shutdown();
//     return 0;
// }



// #include <rclcpp/rclcpp.hpp>
// #include <mongocxx/client.hpp>
// #include <mongocxx/instance.hpp>
// #include <mongocxx/uri.hpp>
// #include <bsoncxx/builder/stream/document.hpp>
// #include <tf2/LinearMath/Quaternion.h>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// #include <fstream>
// #include <sstream>
// #include <string>
// #include <vector>
// #include <stdexcept> // Needed for std::exception

// using bsoncxx::builder::stream::document;
// using bsoncxx::builder::stream::finalize;

// class MongoPoseUploader : public rclcpp::Node
// {
// public:
//     MongoPoseUploader() 
//     : Node("mongo_pose_uploader"),
//       client_(mongocxx::uri("mongodb://localhost:27017"))
//     {
//         db_ = client_["benchmark_db"];
//         coll_ = db_["shared_300_poses"];
        
//         coll_.delete_many({});
//         RCLCPP_INFO(this->get_logger(), "🟢 Connected to MongoDB. Cleared old collection.");

//         std::string csv_path = "/home/ubuntu/metric_ws/workspace/shared_300_poses_for_both_arms.csv";
//         std::ifstream file(csv_path);
        
//         if (!file.is_open()) {
//             RCLCPP_ERROR(this->get_logger(), "❌ Failed to open CSV file: %s", csv_path.c_str());
//             return;
//         }

//         std::string line;
//         std::getline(file, line); // Skip the header row

//         int count = 0;
//         std::vector<bsoncxx::document::value> docs;

//         while (std::getline(file, line)) {
//             std::stringstream ss(line);
//             std::string token;
//             std::vector<std::string> row;
            
//             while (std::getline(ss, token, ',')) {
//                 row.push_back(token);
//             }

//             if (row.size() >= 7) {
//                 try {
//                     // Safe conversion: will throw and be caught if data is invalid (e.g., NaN, empty)
//                     double x = std::stod(row[0]);
//                     double y = std::stod(row[1]);
//                     double z = std::stod(row[2]);
//                     double roll = std::stod(row[3]);
//                     double pitch = std::stod(row[4]);
//                     double yaw = std::stod(row[5]);
//                     std::string source = row[6];

//                     tf2::Quaternion q;
//                     q.setRPY(roll, pitch, yaw); 

//                     auto builder = document{};
//                     bsoncxx::document::value doc_value = builder
//                         << "pose_id" << count + 1
//                         << "x" << x
//                         << "y" << y
//                         << "z" << z
//                         << "qx" << q.x()
//                         << "qy" << q.y()
//                         << "qz" << q.z()
//                         << "qw" << q.w()
//                         << "source" << source
//                         << finalize;

//                     docs.push_back(doc_value);
//                     count++;
//                 } 
//                 catch (const std::exception& e) {
//                     // Catches std::invalid_argument and std::out_of_range
//                     RCLCPP_WARN(this->get_logger(), "⚠️ Skipping malformed CSV row. Error: %s", e.what());
//                     continue; // Skip this row and proceed to the next one
//                 }
//             }
//         }

//         if (!docs.empty()) {
//             coll_.insert_many(docs);
//             RCLCPP_INFO(this->get_logger(), "✅ Success! Batch uploaded %d poses to MongoDB [benchmark_db -> shared_300_poses]", count);
//         } else {
//             RCLCPP_WARN(this->get_logger(), "⚠️ No valid poses found in the CSV to upload.");
//         }
//     }

// private:
//     mongocxx::client client_;
//     mongocxx::database db_;
//     mongocxx::collection coll_;
// };

// int main(int argc, char **argv)
// {
//     rclcpp::init(argc, argv);
//     mongocxx::instance inst{};
//     auto node = std::make_shared<MongoPoseUploader>();
//     rclcpp::shutdown();
//     return 0;
// }