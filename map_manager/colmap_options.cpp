/*
 * @Description: vslam yaml DTOs → COLMAP online SfM options.
 * @Author: che yifan
 * @Date: 2026-09-10
 */
#include "map_manager/colmap_options.h"

#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"

namespace vslam
{

  colmap::FeatureExtractionOptions ToColmap(const SiftExtractConfig &cfg)
  {
    colmap::FeatureExtractionOptions options(colmap::FeatureExtractorType::SIFT);
    options.use_gpu = cfg.use_gpu;
    options.gpu_index = cfg.gpu_index;
    options.max_image_size = cfg.max_image_size;
    options.sift->max_num_features = cfg.max_num_features;
    options.sift->first_octave = cfg.first_octave;
    options.sift->num_octaves = cfg.num_octaves;
    options.sift->octave_resolution = cfg.octave_resolution;
    options.sift->peak_threshold = cfg.peak_threshold;
    options.sift->edge_threshold = cfg.edge_threshold;
    options.sift->estimate_affine_shape = cfg.estimate_affine_shape;
    options.sift->max_num_orientations = cfg.max_num_orientations;
    options.sift->upright = cfg.upright;
    return options;
  }

  colmap::OnlineMatchingOptions ToColmap(const MatchConfig &cfg)
  {
    colmap::OnlineMatchingOptions options;
    options.overlap = cfg.overlap;
    options.spatial_max_distance = cfg.spatial_max_distance;
    options.spatial_max_angle_deg = cfg.spatial_max_angle_deg;
    options.spatial_max_num_images = cfg.spatial_max_num_images;
    options.matching = colmap::FeatureMatchingOptions(
        colmap::FeatureMatcherType::SIFT_BRUTEFORCE);
    options.matching.use_gpu = cfg.use_gpu;
    options.matching.gpu_index = cfg.gpu_index;
    options.matching.max_num_matches = cfg.max_num_matches;
    options.matching.num_threads = cfg.use_gpu ? 1 : cfg.num_threads;
    options.matching.guided_matching = false;
    options.matching.sift->max_ratio = cfg.max_ratio;
    options.matching.sift->max_distance = cfg.max_distance;
    options.matching.sift->cross_check = cfg.cross_check;
    options.matching.sift->cpu_brute_force_matcher = cfg.cpu_brute_force_matcher;
    options.e_only = cfg.e_only;
    options.geometry.min_num_inliers = cfg.min_num_inliers;
    options.geometry.detect_watermark = cfg.detect_watermark;
    options.geometry.use_degensac = cfg.use_degensac;
    options.geometry.ransac_options.max_error = cfg.max_error;
    options.geometry.ransac_options.min_num_trials = cfg.ransac_min_num_trials;
    options.geometry.ransac_options.max_num_trials = cfg.ransac_max_num_trials;
    options.geometry.ransac_options.confidence = cfg.ransac_confidence;
    return options;
  }

  colmap::OnlineMapperOptions ToColmap(const MapperConfig &cfg)
  {
    colmap::OnlineMapperOptions options;
    options.min_num_inliers = cfg.min_num_inliers;
    options.abs_pose_min_num_inliers = cfg.abs_pose_min_num_inliers;
    options.abs_pose_max_error = cfg.abs_pose_max_error;
    options.abs_pose_min_inlier_ratio = cfg.abs_pose_min_inlier_ratio;
    options.ba_local_num_images = cfg.ba_local_num_images;
    options.ba_min_covisibility_points = cfg.ba_min_covisibility_points;
    options.filter_max_reproj_error = cfg.filter_max_reproj_error;
    options.filter_min_tri_angle = cfg.filter_min_tri_angle;
    options.triangulation.max_transitivity = cfg.tri_max_transitivity;
    options.triangulation.create_max_angle_error = cfg.tri_create_max_angle_error;
    options.triangulation.continue_max_angle_error =
        cfg.tri_continue_max_angle_error;
    options.triangulation.merge_max_reproj_error = cfg.tri_merge_max_reproj_error;
    options.triangulation.complete_max_reproj_error =
        cfg.tri_complete_max_reproj_error;
    options.triangulation.complete_max_transitivity =
        cfg.tri_complete_max_transitivity;
    options.triangulation.min_angle = cfg.tri_min_angle;
    options.triangulation.ignore_two_view_tracks = cfg.tri_ignore_two_view_tracks;
    options.local_map_num_images = cfg.local_map_num_images;
    options.local_map_search_radius_px = cfg.local_map_search_radius_px;
    options.local_map_refine_radius_px = cfg.local_map_refine_radius_px;
    options.local_map_max_desc_distance = cfg.local_map_max_desc_distance;
    options.local_map_max_ratio = cfg.local_map_max_ratio;
    options.scale_check_max_ratio = cfg.scale_check_max_ratio;
    options.local_map_spatial_num_images = cfg.local_map_spatial_num_images;
    options.local_map_spatial_max_distance_m =
        cfg.local_map_spatial_max_distance_m;
    options.local_map_spatial_max_angle_deg =
        cfg.local_map_spatial_max_angle_deg;
    options.ba_vio_prior_rot_sigma_deg = cfg.ba_vio_prior_rot_sigma_deg;
    options.ba_vio_prior_trans_sigma_m = cfg.ba_vio_prior_trans_sigma_m;
    options.ba_vio_prior_trans_sigma_ratio = cfg.ba_vio_prior_trans_sigma_ratio;
    options.ba_vio_prior_max_gap = cfg.ba_vio_prior_max_gap;
    options.ba_fixed_min_points = cfg.ba_fixed_min_points;
    options.ba_loop_max_num_iterations = cfg.ba_loop_max_num_iterations;
    options.ba_min_window_images = cfg.ba_min_window_images;
    options.ba_temporal_window_images = cfg.ba_temporal_window_images;
    options.sparse_write_every_n = cfg.sparse_write_every_n;
    options.cull_enabled = cfg.cull_enabled;
    options.cull_redundant_ratio = cfg.cull_redundant_ratio;
    options.cull_min_other_observers = cfg.cull_min_other_observers;
    options.cull_max_scale_ratio = cfg.cull_max_scale_ratio;
    options.cull_min_num_points = cfg.cull_min_num_points;
    options.cull_protect_recent = cfg.cull_protect_recent;
    options.cull_max_neighbor_distance_m = cfg.cull_max_neighbor_distance_m;
    options.cull_max_neighbor_angle_deg = cfg.cull_max_neighbor_angle_deg;
    options.cull_max_per_frame = cfg.cull_max_per_frame;
    options.cull_prior_session = cfg.cull_prior_session;
    options.insert_check_enabled = cfg.insert_check_enabled;
    options.insert_redundant_ratio = cfg.insert_redundant_ratio;
    options.insert_ref_tracked_ratio = cfg.insert_ref_tracked_ratio;
    options.insert_min_tracked = cfg.insert_min_tracked;
    options.insert_max_kf_distance_m = cfg.insert_max_kf_distance_m;
    options.insert_max_kf_angle_deg = cfg.insert_max_kf_angle_deg;
    options.insert_max_consecutive_skips = cfg.insert_max_consecutive_skips;
    options.point_stats_enabled = cfg.point_stats_enabled;
    options.stale_min_visible = cfg.stale_min_visible;
    options.stale_max_found_ratio = cfg.stale_max_found_ratio;
    options.stale_session_decay = cfg.stale_session_decay;
    options.stale_prior_session_only = cfg.stale_prior_session_only;
    options.capacity_enabled = cfg.capacity_enabled;
    options.capacity_voxel_m = cfg.capacity_voxel_m;
    options.capacity_yaw_bins = cfg.capacity_yaw_bins;
    options.capacity_max_per_cell = cfg.capacity_max_per_cell;
    options.capacity_replace_ratio = cfg.capacity_replace_ratio;
    options.capacity_min_session_frames = cfg.capacity_min_session_frames;
    return options;
  }

  colmap::OnlineLoopCloserOptions ToColmap(const LoopCloserConfig &cfg)
  {
    colmap::OnlineLoopCloserOptions options;
    options.enabled = cfg.enabled;
    options.mixvpr_engine_path = cfg.mixvpr_engine_path;
    options.mixvpr_use_gpu = cfg.mixvpr_use_gpu;
    options.mixvpr_gpu_index = cfg.mixvpr_gpu_index;
    options.cooldown_num_images = cfg.cooldown_num_images;
    options.min_frame_gap = cfg.min_frame_gap;
    options.max_distance = cfg.max_distance;
    options.max_view_angle_deg = cfg.max_view_angle_deg;
    options.min_mixvpr_score = cfg.min_mixvpr_score;
    options.topk = cfg.topk;
    options.min_covisibility_points = cfg.min_covisibility_points;
    options.min_candidate_points = cfg.min_candidate_points;
    options.min_num_matches = cfg.min_num_matches;
    options.min_num_verified_matches = cfg.min_num_verified_matches;
    options.min_num_3d_correspondences = cfg.min_num_3d_correspondences;
    options.min_num_3d_inliers = cfg.min_num_3d_inliers;
    options.correct_local = cfg.correct_local;
    options.max_correct_connected = cfg.max_correct_connected;
    options.pnp_max_error = cfg.pnp_max_error;
    options.guided_radius_px = cfg.guided_radius_px;
    options.scale_check_max_ratio = cfg.scale_check_max_ratio;
    options.island_2d3d = cfg.island_2d3d;
    options.island_epipolar_px = cfg.island_epipolar_px;
    options.max_rot_vs_essential_deg = cfg.max_rot_vs_essential_deg;
    options.max_correction_m = cfg.max_correction_m;
    options.max_correction_deg = cfg.max_correction_deg;
    options.pose_graph_min_covis = cfg.pose_graph_min_covis;
    options.pose_graph_4dof = cfg.pose_graph_4dof;
    options.pose_graph_vio_odometry = cfg.pose_graph_vio_odometry;
    options.pose_graph_odom_rot_sigma_deg = cfg.pose_graph_odom_rot_sigma_deg;
    options.pose_graph_odom_trans_sigma_m = cfg.pose_graph_odom_trans_sigma_m;
    options.pose_graph_odom_trans_sigma_ratio = cfg.pose_graph_odom_trans_sigma_ratio;
    options.pose_graph_loop_min_rot_sigma_deg = cfg.pose_graph_loop_min_rot_sigma_deg;
    options.pose_graph_loop_min_trans_sigma_m = cfg.pose_graph_loop_min_trans_sigma_m;
    options.rover_enabled = cfg.rover_enabled;
    options.rover_min_threshold_m = cfg.rover_min_threshold_m;
    options.rover_threshold_ratio = cfg.rover_threshold_ratio;
    options.fuse_radius_px = cfg.fuse_radius_px;
    options.fuse_max_point_distance = cfg.fuse_max_point_distance;
    options.reloc_topk = cfg.reloc_topk;
    options.reloc_min_mixvpr_score = cfg.reloc_min_mixvpr_score;
    options.reloc_min_num_matches = cfg.reloc_min_num_matches;
    options.reloc_min_num_3d_correspondences = cfg.reloc_min_num_3d_correspondences;
    options.reloc_min_inliers = cfg.reloc_min_inliers;
    options.reloc_max_rot_vs_essential_deg = cfg.reloc_max_rot_vs_essential_deg;
    return options;
  }

} // namespace vslam
