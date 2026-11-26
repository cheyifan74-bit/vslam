/*
 * @Description: vslam.yaml → mapping config DTOs → COLMAP online SfM options.
 * @Author: che yifan
 * @Date: 2026-09-10
 */
#include "map_manager/colmap_options.h"

#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"

#include "utility/config_reader.h"
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
    options.filter_two_view_min_tri_angle = cfg.filter_two_view_min_tri_angle;
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
    options.ba_vio_gravity_sigma_deg = cfg.ba_vio_gravity_sigma_deg;
    options.ba_fixed_min_points = cfg.ba_fixed_min_points;
    options.ba_loop_max_num_iterations = cfg.ba_loop_max_num_iterations;
    options.ba_num_threads = cfg.ba_num_threads;
    options.ba_loop_pose_prior_rot_sigma_deg =
        cfg.ba_loop_pose_prior_rot_sigma_deg;
    options.ba_loop_pose_prior_trans_sigma_m =
        cfg.ba_loop_pose_prior_trans_sigma_m;
    options.ba_loop_min_point_track_length = cfg.ba_loop_min_point_track_length;
    options.ba_loop_min_point_tri_angle_deg =
        cfg.ba_loop_min_point_tri_angle_deg;
    options.ba_min_window_images = cfg.ba_min_window_images;
    options.ba_temporal_window_images = cfg.ba_temporal_window_images;
    options.sparse_write_every_n = cfg.sparse_write_every_n;
    options.capacity_enabled = cfg.capacity_enabled;
    options.capacity_voxel_m = cfg.capacity_voxel_m;
    options.capacity_view_angle_deg = cfg.capacity_view_angle_deg;
    options.capacity_max_per_cell = cfg.capacity_max_per_cell;
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
    options.pose_graph_num_threads = cfg.pose_graph_num_threads;
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


  void SiftExtractConfig::load(ConfigReader &reader)
  {
    reader.read("sift_use_gpu", use_gpu);
    reader.read("sift_gpu_index", gpu_index);
    reader.read("sift_max_num_features", max_num_features);
    reader.read("sift_first_octave", first_octave);
    reader.read("sift_num_octaves", num_octaves);
    reader.read("sift_octave_resolution", octave_resolution);
    reader.read("sift_peak_threshold", peak_threshold);
    reader.read("sift_edge_threshold", edge_threshold);
    reader.read("sift_estimate_affine_shape", estimate_affine_shape);
    reader.read("sift_max_num_orientations", max_num_orientations);
    reader.read("sift_upright", upright);
    reader.read("sift_max_image_size", max_image_size);
  }

  void MatchConfig::load(ConfigReader &reader)
  {
    reader.read("match_use_gpu", use_gpu);
    reader.read("match_gpu_index", gpu_index);
    reader.read("match_overlap", overlap);
    reader.read("match_spatial_max_distance", spatial_max_distance);
    reader.read("match_spatial_max_angle_deg", spatial_max_angle_deg);
    reader.read("match_spatial_max_num_images", spatial_max_num_images);
    reader.read("match_num_threads", num_threads);
    reader.read("match_max_num_matches", max_num_matches);
    reader.read("sift_match_max_ratio", max_ratio);
    reader.read("sift_match_max_distance", max_distance);
    reader.read("sift_match_cross_check", cross_check);
    reader.read("match_cpu_brute_force", cpu_brute_force_matcher);
    reader.read("match_min_num_inliers", min_num_inliers);
    reader.read("match_max_error", max_error);
    reader.read("match_tvg_e_only", e_only);
    reader.read("match_detect_watermark", detect_watermark);
    reader.read("match_use_degensac", use_degensac);
    reader.read("match_ransac_min_num_trials", ransac_min_num_trials);
    reader.read("match_ransac_max_num_trials", ransac_max_num_trials);
    reader.read("match_ransac_confidence", ransac_confidence);
  }

  void MapperConfig::load(ConfigReader &reader)
  {
    reader.read("mapper_min_num_inliers", min_num_inliers);
    reader.read("mapper_abs_pose_min_num_inliers", abs_pose_min_num_inliers);
    reader.read("mapper_abs_pose_max_error", abs_pose_max_error);
    reader.read("mapper_abs_pose_min_inlier_ratio", abs_pose_min_inlier_ratio);
    reader.read("mapper_ba_local_num_images", ba_local_num_images);
    reader.read("mapper_ba_min_covisibility_points", ba_min_covisibility_points);
    reader.read("mapper_filter_max_reproj_error", filter_max_reproj_error);
    reader.read("mapper_filter_min_tri_angle", filter_min_tri_angle);
    reader.read("mapper_filter_two_view_min_tri_angle", filter_two_view_min_tri_angle);
    reader.read("mapper_tri_max_transitivity", tri_max_transitivity);
    reader.read("mapper_tri_create_max_angle_error", tri_create_max_angle_error);
    reader.read("mapper_tri_continue_max_angle_error", tri_continue_max_angle_error);
    reader.read("mapper_tri_merge_max_reproj_error", tri_merge_max_reproj_error);
    reader.read("mapper_tri_complete_max_reproj_error", tri_complete_max_reproj_error);
    reader.read("mapper_tri_complete_max_transitivity", tri_complete_max_transitivity);
    reader.read("mapper_tri_min_angle", tri_min_angle);
    reader.read("mapper_tri_ignore_two_view_tracks", tri_ignore_two_view_tracks);
    reader.read("mapper_local_map_num_images", local_map_num_images);
    reader.read("mapper_local_map_search_radius_px", local_map_search_radius_px);
    reader.read("mapper_local_map_refine_radius_px", local_map_refine_radius_px);
    reader.read("mapper_local_map_max_desc_distance", local_map_max_desc_distance);
    reader.read("mapper_local_map_max_ratio", local_map_max_ratio);
    reader.read("mapper_scale_check_max_ratio", scale_check_max_ratio);
    reader.read("mapper_local_map_spatial_num_images", local_map_spatial_num_images);
    reader.read("mapper_local_map_spatial_max_distance_m", local_map_spatial_max_distance_m);
    reader.read("mapper_local_map_spatial_max_angle_deg", local_map_spatial_max_angle_deg);
    reader.read("mapper_ba_vio_prior_rot_sigma_deg", ba_vio_prior_rot_sigma_deg);
    reader.read("mapper_ba_vio_prior_trans_sigma_m", ba_vio_prior_trans_sigma_m);
    reader.read("mapper_ba_vio_prior_trans_sigma_ratio", ba_vio_prior_trans_sigma_ratio);
    reader.read("mapper_ba_vio_prior_max_gap", ba_vio_prior_max_gap);
    reader.read("mapper_ba_vio_gravity_sigma_deg", ba_vio_gravity_sigma_deg);
    reader.read("mapper_ba_fixed_min_points", ba_fixed_min_points);
    reader.read("mapper_ba_loop_max_num_iterations", ba_loop_max_num_iterations);
    reader.read("mapper_ba_num_threads", ba_num_threads);
    reader.read("mapper_ba_loop_pose_prior_rot_sigma_deg", ba_loop_pose_prior_rot_sigma_deg);
    reader.read("mapper_ba_loop_pose_prior_trans_sigma_m", ba_loop_pose_prior_trans_sigma_m);
    reader.read("mapper_ba_loop_min_point_track_length", ba_loop_min_point_track_length);
    reader.read("mapper_ba_loop_min_point_tri_angle_deg", ba_loop_min_point_tri_angle_deg);
    reader.read("mapper_ba_min_window_images", ba_min_window_images);
    reader.read("mapper_ba_temporal_window_images", ba_temporal_window_images);
    reader.read("mapper_sparse_write_every_n", sparse_write_every_n);
    reader.read("mapper_pending_max_frames", pending_max_frames);
    reader.read("mapper_capacity_enabled", capacity_enabled);
    reader.read("mapper_capacity_voxel_m", capacity_voxel_m);
    reader.read("mapper_capacity_view_angle_deg", capacity_view_angle_deg);
    reader.read("mapper_capacity_max_per_cell", capacity_max_per_cell);
    reader.read("map_compact_database_on_exit", compact_database_on_exit);
  }

  void LoopCloserConfig::load(ConfigReader &reader)
  {
    reader.read("loop_enabled", enabled);
    reader.read("loop_mixvpr_engine_path", mixvpr_engine_path);
    reader.read("loop_mixvpr_use_gpu", mixvpr_use_gpu);
    reader.read("loop_mixvpr_gpu_index", mixvpr_gpu_index);
    reader.read("loop_cooldown_num_images", cooldown_num_images);
    reader.read("loop_min_frame_gap", min_frame_gap);
    reader.read("loop_max_distance", max_distance);
    reader.read("loop_max_view_angle_deg", max_view_angle_deg);
    reader.read("loop_min_mixvpr_score", min_mixvpr_score);
    reader.read("loop_topk", topk);
    reader.read("loop_min_covisibility_points", min_covisibility_points);
    reader.read("loop_min_candidate_points", min_candidate_points);
    reader.read("loop_min_num_matches", min_num_matches);
    reader.read("loop_min_num_verified_matches", min_num_verified_matches);
    reader.read("loop_min_num_3d_correspondences", min_num_3d_correspondences);
    reader.read("loop_min_num_3d_inliers", min_num_3d_inliers);
    reader.read("loop_correct_local", correct_local);
    reader.read("loop_max_correct_connected", max_correct_connected);
    reader.read("loop_pnp_max_error", pnp_max_error);
    reader.read("loop_guided_radius_px", guided_radius_px);
    reader.read("loop_scale_check_max_ratio", scale_check_max_ratio);
    reader.read("loop_island_2d3d", island_2d3d);
    reader.read("loop_island_epipolar_px", island_epipolar_px);
    reader.read("loop_max_rot_vs_essential_deg", max_rot_vs_essential_deg);
    reader.read("loop_max_correction_m", max_correction_m);
    reader.read("loop_max_correction_deg", max_correction_deg);
    reader.read("loop_pose_graph_min_covis", pose_graph_min_covis);
    reader.read("loop_pose_graph_4dof", pose_graph_4dof);
    reader.read("loop_pose_graph_vio_odometry", pose_graph_vio_odometry);
    reader.read("loop_pose_graph_odom_rot_sigma_deg", pose_graph_odom_rot_sigma_deg);
    reader.read("loop_pose_graph_odom_trans_sigma_m", pose_graph_odom_trans_sigma_m);
    reader.read("loop_pose_graph_odom_trans_sigma_ratio", pose_graph_odom_trans_sigma_ratio);
    reader.read("loop_pose_graph_loop_min_rot_sigma_deg", pose_graph_loop_min_rot_sigma_deg);
    reader.read("loop_pose_graph_loop_min_trans_sigma_m", pose_graph_loop_min_trans_sigma_m);
    reader.read("loop_pose_graph_num_threads", pose_graph_num_threads);
    reader.read("loop_rover_enabled", rover_enabled);
    reader.read("loop_rover_min_threshold_m", rover_min_threshold_m);
    reader.read("loop_rover_threshold_ratio", rover_threshold_ratio);
    reader.read("loop_fuse_radius_px", fuse_radius_px);
    reader.read("loop_fuse_max_point_distance", fuse_max_point_distance);
    reader.read("loop_reloc_topk", reloc_topk);
    reader.read("loop_reloc_min_mixvpr_score", reloc_min_mixvpr_score);
    reader.read("loop_reloc_min_num_matches", reloc_min_num_matches);
    reader.read("loop_reloc_min_num_3d_correspondences", reloc_min_num_3d_correspondences);
    reader.read("loop_reloc_min_inliers", reloc_min_inliers);
    reader.read("loop_reloc_max_rot_vs_essential_deg", reloc_max_rot_vs_essential_deg);
  }

} // namespace vslam
