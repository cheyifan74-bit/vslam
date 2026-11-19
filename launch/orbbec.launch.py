# ================================================================
# Orbbec Gemini 335L (双目 IR + IMU) + VSLAM 一键启动
#
# 使用前:
#   source /opt/ros/humble/setup.bash
#   source ~/vslam_ws/install/setup.bash
#   ros2 launch vslam orbbec.launch.py
#
# 只启动相机(不跑 vslam):
#   ros2 launch vslam orbbec.launch.py docam:=true dovslam:=false
#
# 只启动 vslam(相机已在别的终端跑):
#   ros2 launch vslam orbbec.launch.py docam:=false
#
# 话题:
#   左目 /camera/left_ir/image_raw     (mono8 @640x480 30Hz)  <- VIO 输入
#   右目 /camera/right_ir/image_raw    (mono8 @640x480 30Hz)  <- VIO 输入
#   彩色 /camera/color/image_raw       (rgb8  @640x480 30Hz)
#   深度 /camera/depth/image_raw       (Y16, HW D2C 对齐到 Color)
#   IMU  /camera/gyro_accel/sample     (sensor_msgs/Imu, 200Hz)
#
# 说明:
#   - Gemini 335L 只有一个 COLOR 传感器 + 一对 LEFT_IR/RIGHT_IR,
#     所以"RGB 双目"硬件上不存在; 双目 VIO 只能用 IR 对。
#   - 左右 IR 内参相同(硬件已立体校正), baseline 95.352 mm。
#   - 640x480 实测 29.5Hz 稳定; 1280x800 会掉帧。
#
# 关于红外散斑 与 "既要双目又要深度":
#   散斑来自相机自带的主动红外激光器, 而深度又依赖它。
#
#     laser:=off         深度差, IR 干净                         <- 默认
#     laser:=low / full  深度好, IR 有散斑
#     laser:=interleave  硬件交替: 偶帧开激光(深度好) / 奇帧关激光(IR 干净)
#                        需固件 >= 1.4.00 (335L 已满足)。有效帧率约减半(~15Hz)。
#                        默认把 IR 滤成关激光帧、Depth 滤成开激光帧。
#
# 深度来源:
#   要"直接用相机深度图"就用 /camera/depth/image_raw (Y16),
#   此时可把 vslam.yaml 的 dense_enabled 设 false —— DenseMapper 的 S2M2
#   会从双目另算深度, 与相机深度重复。
#
# 用法:
#   ros2 launch vslam orbbec.launch.py
#   ros2 launch vslam orbbec.launch.py laser:=full
#   ros2 launch vslam orbbec.launch.py laser:=interleave
# ================================================================
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_setup(context):
    vslam_share = get_package_share_directory("vslam")
    orbbec_share = get_package_share_directory("orbbec_camera")

    config = LaunchConfiguration("config").perform(context)
    # 优先读源码树, 这样新增配置不用重编译 vslam 包
    src_config = os.path.expanduser(
        "~/vslam_ws/src/vslam/config/{}/estimator_config.yaml".format(config))
    config_path = src_config if os.path.isfile(src_config) else os.path.join(
        vslam_share, "config", config, "estimator_config.yaml")

    rviz_config = LaunchConfiguration("rviz_config").perform(context)
    if not rviz_config:
        rviz_config = os.path.join(vslam_share, "launch", "display_ros2.rviz")

    # laser: off / low / full / interleave (兼容 true/false)
    laser = LaunchConfiguration("laser").perform(context).strip().lower()
    laser_off = laser in ("off", "false", "0", "none")
    laser_interleave = laser in ("interleave", "alt", "onoff", "on-off", "laser_on_off")
    # low -> 功率 1, full/interleave -> 功率 6
    laser_level = 6 if laser in ("full", "high", "6", "on", "true") or laser_interleave else 1

    cam_args = {
        "camera_name": LaunchConfiguration("camera_name"),
        # Color 640x480: 335L 硬件 D2C 支持的分辨率 (1280x720 只能软件 D2C)
        "enable_color": LaunchConfiguration("color"),
        "color_width": "640",
        "color_height": "480",
        "color_fps": "30",
        "color_format": "RGB",
        # depth 必须开: 结构光/激光靠它点亮, 否则 IR 全黑
        "enable_depth": "true",
        "enable_left_ir": "true",
        "enable_right_ir": "true",
        "enable_ir": "false",
        # 激光/散斑: off=关, low=功率1, full=功率6, interleave=硬件交替开/关
        "enable_laser": "false" if laser_off else "true",
        "enable_ldp": "false" if laser_off else "true",
        # 硬件 Depth→Color 对齐后, /camera/depth/image_raw 与 Color 像素对齐
        "depth_registration": "true",
        "align_mode": "HW",
        # HW D2C 后可发对齐到 Color 的点云, 便于 RViz 查看
        "enable_point_cloud": "true",
        "enable_colored_point_cloud": "true",
        "depth_width": "640",
        "depth_height": "480",
        "depth_fps": "30",
        "left_ir_width": "640",
        "left_ir_height": "480",
        "left_ir_fps": "30",
        "left_ir_format": "Y8",
        "right_ir_width": "640",
        "right_ir_height": "480",
        "right_ir_fps": "30",
        "right_ir_format": "Y8",
        "enable_accel": "true",
        "enable_gyro": "true",
        "accel_rate": "200hz",
        "gyro_rate": "200hz",
        "enable_sync_output_accel_gyro": "true",
        "enable_frame_sync": "true",
        "enable_publish_extrinsic": "false",
        # 只发 raw, 关掉 compressed/theora 省带宽
        "left_ir.image_raw.enable_pub_plugins": '["image_transport/raw"]',
        "right_ir.image_raw.enable_pub_plugins": '["image_transport/raw"]',
        "color.image_raw.enable_pub_plugins": '["image_transport/raw"]',
        "depth.image_raw.enable_pub_plugins": '["image_transport/raw"]',
    }
    if not laser_off:
        cam_args["laser_energy_level"] = str(laser_level)
        cam_args["ldp_power_level"] = str(laser_level)
    if laser_interleave:
        # Gemini 330 "Laser On-Off": index0=开激光(散斑/深度好), index1=关激光(IR 干净)
        cam_args.update({
            "interleave_ae_mode": "laser",
            "interleave_frame_enable": "true",
            "laser_index0_laser_control": "1",
            "laser_index1_laser_control": "0",
            # VIO 用干净 IR (关激光帧 sequence_id=1)
            "enable_left_ir_sequence_id_filter": "true",
            "left_ir_sequence_id_filter_id": "1",
            "enable_right_ir_sequence_id_filter": "true",
            "right_ir_sequence_id_filter_id": "1",
            # 深度用开激光帧 (sequence_id=0)
            "enable_sequence_id_filter": "true",
            "sequence_id_filter_id": "0",
        })

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(orbbec_share, "launch", "gemini_330_series.launch.py")),
        condition=IfCondition(LaunchConfiguration("docam")),
        launch_arguments=cam_args.items(),
    )

    vslam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(vslam_share, "launch", "vslam.launch.py")),
        condition=IfCondition(LaunchConfiguration("dovslam")),
        launch_arguments={
            "config": config,
            "config_path": config_path,
            "doviz": LaunchConfiguration("doviz"),
            "rviz_config": rviz_config,
            "verbosity": LaunchConfiguration("verbosity"),
            "namespace": LaunchConfiguration("namespace"),
        }.items(),
    )

    return [camera, vslam]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value="orbbec"),
        DeclareLaunchArgument("camera_name", default_value="camera"),
        DeclareLaunchArgument("namespace", default_value="ov_msckf"),
        DeclareLaunchArgument("verbosity", default_value="INFO"),
        DeclareLaunchArgument("docam", default_value="true"),
        DeclareLaunchArgument("dovslam", default_value="true"),
        DeclareLaunchArgument("doviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=""),
        # 原始 RGB 是否发布 (不影响 VIO 用哪路)
        DeclareLaunchArgument("color", default_value="true"),
        # 主动红外激光: off / low / full / interleave(硬件帧交错开/关)
        DeclareLaunchArgument("laser", default_value="off"),
        OpaqueFunction(function=launch_setup),
    ])
