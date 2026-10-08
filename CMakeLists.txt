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

# Eigen >= 3.4 for COLMAP 4.x (vendored).
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

# Synthetic OpenImageIOConfig for COLMAP (Ubuntu package has none).
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

# COLMAP options (CUDA GPU SIFT; GUI/OpenGL off).
set(CUDA_ENABLED ON CACHE BOOL "" FORCE)
set(GUI_ENABLED OFF CACHE BOOL "" FORCE)
set(OPENGL_ENABLED OFF CACHE BOOL "" FORCE)
set(ONNX_ENABLED ON CACHE BOOL "" FORCE)
set(MVS_ENABLED OFF CACHE BOOL "" FORCE)
set(CGAL_ENABLED OFF CACHE BOOL "" FORCE)
set(LSD_ENABLED OFF CACHE BOOL "" FORCE)
set(DOWNLOAD_ENABLED OFF CACHE BOOL "" FORCE)
set(TESTS_ENABLED OFF CACHE BOOL "" FORCE)
set(UNINSTALL_ENABLED OFF CACHE BOOL "" FORCE)
set(FETCH_ONNX ON CACHE BOOL "" FORCE)
set(FETCH_POSELIB ON CACHE BOOL "" FORCE)
set(FETCH_FAISS ON CACHE BOOL "" FORCE)
set(COLMAP_HASH_MAP_BACKEND "STD" CACHE STRING "" FORCE)
set(CCACHE_ENABLED ON CACHE BOOL "" FORCE)

add_subdirectory(colmap EXCLUDE_FROM_ALL)

add_subdirectory(mixvpr)
# Prefer inject over linking vslam_mixvpr into colmap_online_sfm (EXPORT).
if(TARGET colmap_online_sfm AND TARGET vslam_mixvpr)
  target_include_directories(colmap_online_sfm PRIVATE
    $<TARGET_PROPERTY:vslam_mixvpr,INTERFACE_INCLUDE_DIRECTORIES>)
  target_compile_definitions(colmap_online_sfm PRIVATE
    $<TARGET_PROPERTY:vslam_mixvpr,INTERFACE_COMPILE_DEFINITIONS>)
endif()

# Shrink PoseLib/faiss objects (-g3 from parent flags is huge).
if(TARGET PoseLib)
  target_compile_options(PoseLib PRIVATE -g0)
endif()
if(TARGET faiss)
  target_compile_options(faiss PRIVATE -g0)
endif()

# OIIO 2.1 lacks OIIO_MAKE_VERSION (function-like -D; commas break compile_definitions).
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
# Keep vslam_mixvpr against --as-needed (online_sfm needs MixVPR symbols).
if(TARGET vslam_mixvpr)
  target_link_libraries(vslam_node
    -Wl,--no-as-needed
    vslam_mixvpr
    -Wl,--as-needed)
endif()

# Prefer FetchContent ORT 1.27.1 over /usr/local (DT_RPATH beats LD_LIBRARY_PATH).
if(DEFINED onnxruntime_LIBRARY_DIR_HINTS AND onnxruntime_LIBRARY_DIR_HINTS
   AND EXISTS "${onnxruntime_LIBRARY_DIR_HINTS}/libonnxruntime.so.1.27.1")
  set(_ort_lib_dir "${onnxruntime_LIBRARY_DIR_HINTS}")
  set(_ort_rpath "${_ort_lib_dir}")
  message(STATUS "vslam: ONNX Runtime RPATH -> ${_ort_lib_dir}")

  # cuDNN 9 for ORT CUDA EP.
  set(_cudnn9_hints)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c
              "import pathlib, nvidia.cudnn; print(pathlib.Path(nvidia.cudnn.__file__).parent / 'lib')"
      OUTPUT_VARIABLE _py_cudnn_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(_py_cudnn_dir)
      list(APPEND _cudnn9_hints "${_py_cudnn_dir}")
    endif()
  endif()
  list(APPEND _cudnn9_hints
    "$ENV{HOME}/.local/lib/python3.8/site-packages/nvidia/cudnn/lib"
    "$ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/cudnn/lib"
    "$ENV{HOME}/anaconda3/lib/python3.10/site-packages/nvidia/cudnn/lib"
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu)
  find_file(VSLAM_CUDNN9_LIB libcudnn.so.9 HINTS ${_cudnn9_hints})
  if(VSLAM_CUDNN9_LIB)
    get_filename_component(VSLAM_CUDNN9_DIR "${VSLAM_CUDNN9_LIB}" DIRECTORY)
    list(APPEND _ort_rpath "${VSLAM_CUDNN9_DIR}")
    message(STATUS "ORT CUDA EP: cuDNN 9 RPATH -> ${VSLAM_CUDNN9_DIR}")
  else()
    message(WARNING
      "ORT CUDA EP needs libcudnn.so.9. Install nvidia-cudnn-cu12 or set "
      "LD_LIBRARY_PATH; otherwise ONNX CUDA EP may fall back to CPU.")
  endif()

  set_target_properties(vslam_node PROPERTIES BUILD_RPATH "${_ort_rpath}")
  set_property(TARGET vslam_node APPEND_STRING PROPERTY
               LINK_FLAGS " -Wl,--disable-new-dtags")

  if(TARGET colmap_feature)
    set_target_properties(colmap_feature PROPERTIES BUILD_RPATH "${_ort_rpath}")
    set_property(TARGET colmap_feature APPEND_STRING PROPERTY
                 LINK_FLAGS " -Wl,--disable-new-dtags")
  endif()
endif()

install(TARGETS vslam_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
