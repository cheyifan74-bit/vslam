# Online incremental SfM + loop closing on top of COLMAP (namespace colmap).
# Relies on vslam's COLMAP fork patches (database_session, MixVPR table, ...),
# so it needs the bundled colmap/ and cannot build against upstream COLMAP.
add_library(vslam_online_sfm STATIC
  online_feature_extractor.cc
  online_feature_matcher.cc
  online_incremental_mapper.cc
  online_loop_closer.cc
  online_loop_correction.cc
  online_loop_detection.cc
  online_loop_verify.cc
  online_relocalizer.cc
  online_mapper.cc
  online_pose_graph.cc
  timing_stats.cc
)

target_include_directories(vslam_online_sfm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(vslam_online_sfm
  PUBLIC
    colmap_sfm
    colmap_scene
    colmap_feature
    colmap_estimators
    colmap_geometry
    colmap_util  # Ceres comes in through colmap_estimators.
  PRIVATE
    colmap_sensor
    vslam_mixvpr
)
