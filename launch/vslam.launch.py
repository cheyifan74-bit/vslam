# ================================================================
# VSLAM 在线运行 launch 文件（ROS2，对应 ROS1 的 vslam.launch）
# 启动: ros2 launch vslam vslam.launch.py
#       ros2 launch vslam vslam.launch.py config:=euroc
#       ros2 launch vslam vslam.launch.py config:=euroc dobag:=true bag:=/path/to/V1_01_easy
# 话题来自 config/<config>/kalibr_*.yaml；ROS1 .bag 需先用 rosbags-convert 转成 rosbag2
# 节点放在 namespace "ov_msckf" 下，因此 OpenVINS 话题为 /ov_msckf/*，与 ROS1 一致
# ================================================================
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

launch_args = [
    # 日志级别: ALL, DEBUG, INFO, WARNING, ERROR, SILENT
    DeclareLaunchArgument("verbosity", default_value="INFO"),
    # 配置名（config/<config>/estimator_config.yaml，同目录下还有 kalibr_*.yaml 与 vslam.yaml）
    DeclareLaunchArgument("config", default_value="tum_vi", description="euroc, mynteye, tum_vi"),
    DeclareLaunchArgument("config_path", default_value="",
                          description="estimator_config.yaml 路径；为空则由 config 推导"),
    DeclareLaunchArgument("namespace", default_value="ov_msckf"),
    # 是否启动 RViz2
    DeclareLaunchArgument("doviz", default_value="true"),
    DeclareLaunchArgument("rviz_config", default_value=""),
    # 是否同时回放 rosbag2
    DeclareLaunchArgument("dobag", default_value="false"),
    DeclareLaunchArgument("bag", default_value=os.path.expanduser("~/datasets/euroc/V1_01_easy")),
    DeclareLaunchArgument("bag_start", default_value="0"),
    DeclareLaunchArgument("bag_rate", default_value="1.0"),
    DeclareLaunchArgument("bag_delay", default_value="1"),
]


def launch_setup(context):
    share_dir = get_package_share_directory("vslam")

    config_path = LaunchConfiguration("config_path").perform(context)
    if not config_path:
        config = LaunchConfiguration("config").perform(context)
        config_path = os.path.join(share_dir, "config", config, "estimator_config.yaml")
    if not os.path.isfile(config_path):
        return [LogInfo(msg="ERROR: config file '{}' does not exist - not starting vslam".format(config_path))]

    rviz_config = LaunchConfiguration("rviz_config").perform(context)
    if not rviz_config:
        rviz_config = os.path.join(share_dir, "launch", "display_ros2.rviz")

    # Keyframe 参数请改 config/<config>/vslam.yaml，勿用 ROS param
    vslam_node = Node(
        package="vslam",
        executable="vslam_node",
        name="ov_msckf",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[
            {"verbosity": LaunchConfiguration("verbosity")},
            {"config_path": config_path},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        condition=IfCondition(LaunchConfiguration("doviz")),
        output="log",
        arguments=["-d", rviz_config, "--ros-args", "--log-level", "warn"],
    )

    bag_play = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration("dobag")),
        cmd=[
            "ros2", "bag", "play", LaunchConfiguration("bag"),
            "--rate", LaunchConfiguration("bag_rate"),
            "--start-offset", LaunchConfiguration("bag_start"),
            "--delay", LaunchConfiguration("bag_delay"),
        ],
        output="screen",
    )

    return [vslam_node, rviz_node, bag_play]


def generate_launch_description():
    ld = LaunchDescription(launch_args)
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
