cmake_minimum_required(VERSION 3.3)
project(vslam)

# C++14 required
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable ROS (required by OpenVINS headers: ROS_AVAILABLE guards)
add_definitions(-DROS_AVAILABLE=1)

# Optimization flags (match OpenVINS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g3 -Wall")

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
  message_filters
  image_transport
  tf
  std_msgs
  nav_msgs
)

add_subdirectory(open_vins/ov_core)
add_subdirectory(open_vins/ov_init)
add_subdirectory(open_vins/ov_msckf)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    sensor_msgs
    cv_bridge
    message_filters
    image_transport
    ov_core
    ov_msckf
    ov_init
  INCLUDE_DIRS
    .
)

# Include dirs for our own headers and OpenVINS
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# ============================================================================
# vslam_node executable
# ============================================================================

add_executable(vslam_node
  app/application.cpp
  ros_wrapper/ros_wrapper.cpp
)

target_link_libraries(vslam_node
  ov_msckf_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
)

install(TARGETS vslam_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install config and launch files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
