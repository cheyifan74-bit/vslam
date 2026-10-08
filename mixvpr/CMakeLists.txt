add_library(vslam_mixvpr
  mixvpr_encoder.cc
  mixvpr_encoder.h
  mixvpr_preprocess.h
)

target_include_directories(vslam_mixvpr
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../colmap/src
)

target_link_libraries(vslam_mixvpr
  PUBLIC
    colmap_util
    colmap_sensor
  PRIVATE
    Eigen3::Eigen
)

set(_trt_root_hints "")
if(DEFINED TensorRT_ROOT AND NOT "${TensorRT_ROOT}" STREQUAL "")
  list(APPEND _trt_root_hints "${TensorRT_ROOT}")
endif()
if(DEFINED ENV{TensorRT_ROOT} AND NOT "$ENV{TensorRT_ROOT}" STREQUAL "")
  list(APPEND _trt_root_hints "$ENV{TensorRT_ROOT}")
endif()
if(DEFINED ENV{TENSORRT_ROOT} AND NOT "$ENV{TENSORRT_ROOT}" STREQUAL "")
  list(APPEND _trt_root_hints "$ENV{TENSORRT_ROOT}")
endif()
list(APPEND _trt_root_hints
     /usr
     /usr/local
     /usr/local/tensorrt
     /opt/tensorrt
     /opt/TensorRT)

find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS ${_trt_root_hints}
    PATH_SUFFIXES
        include
        include/x86_64-linux-gnu
        usr/include
        usr/include/x86_64-linux-gnu)
find_library(TensorRT_LIBRARY nvinfer
    HINTS ${_trt_root_hints}
    PATH_SUFFIXES
        lib
        lib64
        lib/x86_64-linux-gnu
        usr/lib
        usr/lib/x86_64-linux-gnu)
find_library(TensorRT_PLUGIN_LIBRARY nvinfer_plugin
    HINTS ${_trt_root_hints}
    PATH_SUFFIXES
        lib
        lib64
        lib/x86_64-linux-gnu
        usr/lib
        usr/lib/x86_64-linux-gnu)

set(_mixvpr_cuda_ok FALSE)
if(TARGET CUDA::cudart)
  set(_mixvpr_cuda_ok TRUE)
else()
  find_package(CUDAToolkit QUIET)
  if(TARGET CUDA::cudart)
    set(_mixvpr_cuda_ok TRUE)
  endif()
endif()

if(TensorRT_INCLUDE_DIR AND TensorRT_LIBRARY AND _mixvpr_cuda_ok)
  message(STATUS "vslam_mixvpr TensorRT: ${TensorRT_LIBRARY}")
  message(STATUS "vslam_mixvpr TensorRT include: ${TensorRT_INCLUDE_DIR}")
  enable_language(CUDA)
  target_include_directories(vslam_mixvpr PRIVATE ${TensorRT_INCLUDE_DIR})
  target_link_libraries(vslam_mixvpr PRIVATE ${TensorRT_LIBRARY} CUDA::cudart)
  if(TensorRT_PLUGIN_LIBRARY)
    target_link_libraries(vslam_mixvpr PRIVATE ${TensorRT_PLUGIN_LIBRARY})
  endif()
  target_compile_definitions(vslam_mixvpr
    PUBLIC
      VSLAM_MIXVPR_TENSORRT_ENABLED
      VSLAM_MIXVPR_CUDA_ENABLED)
  set_source_files_properties(mixvpr_preprocess.cu PROPERTIES LANGUAGE CUDA)
  target_sources(vslam_mixvpr PRIVATE mixvpr_preprocess.cu)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set_target_properties(vslam_mixvpr PROPERTIES CUDA_ARCHITECTURES "75;86;89")
  endif()
  set_target_properties(vslam_mixvpr PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
else()
  message(WARNING
    "vslam_mixvpr: TensorRT/CUDA not found; MixVPR encoder will fail at runtime. "
    "Install system TensorRT or set TensorRT_ROOT.")
endif()
