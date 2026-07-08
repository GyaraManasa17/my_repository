cmake_minimum_required(VERSION 3.8)
project(database_storage)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------
# 1. ROS 2 Dependencies
# -----------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(moveit_ros_warehouse REQUIRED)
find_package(warehouse_ros REQUIRED)

find_package(moveit_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)

# -----------------------------
# 2. MongoDB Dependencies
# -----------------------------
find_package(mongocxx REQUIRED)
find_package(bsoncxx REQUIRED)

# -----------------------------
# 3. Executables
# -----------------------------
add_executable(store src/store.cpp)
add_executable(store_joints src/storing_joints.cpp)

# -----------------------------
# 4. MongoDB linking
# -----------------------------
target_link_libraries(store
  mongo::mongocxx_shared
  mongo::bsoncxx_shared
)

target_link_libraries(store_joints
  mongo::mongocxx_shared
  mongo::bsoncxx_shared
)

# -----------------------------
# 5. ROS Dependencies
# -----------------------------
ament_target_dependencies(store
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  moveit_ros_warehouse
  warehouse_ros
  moveit_msgs
  geometry_msgs
  shape_msgs
)

ament_target_dependencies(store_joints
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  moveit_ros_warehouse
  warehouse_ros
  moveit_msgs
  geometry_msgs
  shape_msgs
)

# -----------------------------
# 6. Install Targets
# -----------------------------
install(TARGETS
  store
  store_joints
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()