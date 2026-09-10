cmake_minimum_required(VERSION 3.12)
project(vslam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DROS_AVAILABLE=1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g3 -Wall")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

# COLMAP 4.x needs Eigen >= 3.4. Point both OpenVINS and COLMAP at vendored headers.
# Config files are generated into the build tree (not a source cmake/ folder).
set(_VSLAM_EIGEN34 "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen-3.4.0")
if(NOT EXISTS "${_VSLAM_EIGEN34}/Eigen/Core")
  message(FATAL_ERROR "Eigen 3.4.0 missing: ${_VSLAM_EIGEN34}")
endif()

set(_VSLAM_DEP_CFG "${CMAKE_BINARY_DIR}/vslam_dep_config")
file(MAKE_DIRECTORY "${_VSLAM_DEP_CFG}")
file(WRITE "${_VSLAM_DEP_CFG}/Eigen3Config.cmake"
"if(NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES \"${_VSLAM_EIGEN34}\")
endif()
set(Eigen3_FOUND TRUE)
set(EIGEN3_FOUND TRUE)
set(EIGEN3_INCLUDE_DIR \"${_VSLAM_EIGEN34}\")
set(EIGEN3_INCLUDE_DIRS \"${_VSLAM_EIGEN34}\")
set(Eigen3_VERSION \"3.4.0\")
")

# Ubuntu OpenImageIO has no CMake package; create one for COLMAP's find_package.
find_path(_OIIO_INC OpenImageIO/imageio.h REQUIRED)
find_library(_OIIO_LIB OpenImageIO REQUIRED)
find_library(_OIIO_UTIL_LIB OpenImageIO_Util)
set(_OIIO_CFG
"if(NOT TARGET OpenImageIO::OpenImageIO)
  add_library(OpenImageIO::OpenImageIO UNKNOWN IMPORTED)
  set_target_properties(OpenImageIO::OpenImageIO PROPERTIES
    IMPORTED_LOCATION \"${_OIIO_LIB}\"
    INTERFACE_INCLUDE_DIRECTORIES \"${_OIIO_INC}\")
endif()
set(OpenImageIO_FOUND TRUE)
set(OpenImageIO_INCLUDE_DIRS \"${_OIIO_INC}\")
set(OpenImageIO_LIBRARIES \"${_OIIO_LIB}\")
")
if(_OIIO_UTIL_LIB)
  string(APPEND _OIIO_CFG
"if(NOT TARGET OpenImageIO::OpenImageIO_Util)
  add_library(OpenImageIO::OpenImageIO_Util UNKNOWN IMPORTED)
  set_target_properties(OpenImageIO::OpenImageIO_Util PROPERTIES
    IMPORTED_LOCATION \"${_OIIO_UTIL_LIB}\"
    INTERFACE_INCLUDE_DIRECTORIES \"${_OIIO_INC}\")
  target_link_libraries(OpenImageIO::OpenImageIO INTERFACE OpenImageIO::OpenImageIO_Util)
endif()
")
endif()
file(WRITE "${_VSLAM_DEP_CFG}/OpenImageIOConfig.cmake" "${_OIIO_CFG}")

set(Eigen3_DIR "${_VSLAM_DEP_CFG}" CACHE PATH "" FORCE)
set(OpenImageIO_DIR "${_VSLAM_DEP_CFG}" CACHE PATH "" FORCE)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time graph program_options)
find_package(SQLite3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
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

# COLMAP submodule (standard add_subdirectory). Extra features off.
# GPU SIFT needs COLMAP_GPU_ENABLED. GUI-off COLMAP also disables OpenGL, so
# CUDA is the only GPU SIFT path. FindDependencies turns CUDA off if missing.
set(CUDA_ENABLED ON CACHE BOOL "" FORCE)
set(GUI_ENABLED OFF CACHE BOOL "" FORCE)
set(OPENGL_ENABLED OFF CACHE BOOL "" FORCE)
set(ONNX_ENABLED OFF CACHE BOOL "" FORCE)
# Also skips thirdparty PoissonRecon (dense meshing; unused by incremental SfM).
set(MVS_ENABLED OFF CACHE BOOL "" FORCE)
set(CGAL_ENABLED OFF CACHE BOOL "" FORCE)
set(LSD_ENABLED OFF CACHE BOOL "" FORCE)
set(DOWNLOAD_ENABLED OFF CACHE BOOL "" FORCE)
set(TESTS_ENABLED OFF CACHE BOOL "" FORCE)
set(UNINSTALL_ENABLED OFF CACHE BOOL "" FORCE)
set(FETCH_ONNX OFF CACHE BOOL "" FORCE)
set(FETCH_POSELIB ON CACHE BOOL "" FORCE)
set(FETCH_FAISS ON CACHE BOOL "" FORCE)
set(COLMAP_HASH_MAP_BACKEND "STD" CACHE STRING "" FORCE)
set(CCACHE_ENABLED ON CACHE BOOL "" FORCE)

add_subdirectory(colmap EXCLUDE_FROM_ALL)

# Parent CMAKE_CXX_FLAGS has -g3; PoseLib/faiss templates then emit huge objects
# (qep.cc.o ~370MB). vslam_node keeps -g3.
if(TARGET PoseLib)
  target_compile_options(PoseLib PRIVATE -g0)
endif()
if(TARGET faiss)
  target_compile_options(faiss PRIVATE -g0)
endif()

# Ubuntu OIIO 2.1 lacks OIIO_MAKE_VERSION. Function-like -D (not compile_definitions:
# CMake would split on commas). Does not modify COLMAP sources.
set(_OIIO_MAKE_VERSION
    "OIIO_MAKE_VERSION(major,minor,patch)=(10000*(major)+100*(minor)+(patch))")
if(TARGET colmap_util)
  target_compile_options(colmap_util PRIVATE "-D${_OIIO_MAKE_VERSION}")
endif()
if(TARGET colmap_sensor)
  target_compile_options(colmap_sensor PRIVATE "-D${_OIIO_MAKE_VERSION}")
endif()

catkin_package(
  CATKIN_DEPENDS
    roscpp
    sensor_msgs
    geometry_msgs
    cv_bridge
    message_filters
    image_transport
    ov_core
    ov_msckf
    ov_init
  INCLUDE_DIRS
    .
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/open_vins/ov_core/src
  ${CMAKE_CURRENT_SOURCE_DIR}/open_vins/ov_init/src
  ${CMAKE_CURRENT_SOURCE_DIR}/open_vins/ov_msckf/src
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_executable(vslam_node
  app/application.cpp
  ros_wrapper/ros_wrapper.cpp
  keyframe_select/keyframe_selector.cpp
  keyframe_select/keyframe_select.cpp
  map_manager/map.cpp
  map_manager/map_manager.cpp
  map_manager/colmap_options.cpp
  utility/math_utils.cpp
)

target_link_libraries(vslam_node
  ov_msckf_lib
  colmap
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
)

install(TARGETS vslam_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
