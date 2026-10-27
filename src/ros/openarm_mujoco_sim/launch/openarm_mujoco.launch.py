"""OpenArm v1.0 bimanual in MuJoCo, driven by ros2_control.

Based on mujoco_ros2_control_demos/01_basic_robot.launch.py.

Usage:
    ros2 launch openarm_mujoco_sim openarm_mujoco.launch.py
    ros2 launch openarm_mujoco_sim openarm_mujoco.launch.py headless:=true

Move the left arm (7 joint angles in radians):
    ros2 topic pub --once /left_forward_position_controller/commands \
      std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0]}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue
from launch_ros.substitutions import FindPackageShare

CONTROLLERS = [
    "joint_state_broadcaster",
    "left_gripper_controller",
    "right_gripper_controller",
]


def launch_setup(context, *args, **kwargs):
    pkg_share = FindPackageShare("openarm_mujoco_sim")

    robot_description_str = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([pkg_share, "urdf", "openarm_mujoco.urdf.xacro"]),
            " headless:=",
            LaunchConfiguration("headless"),
        ]
    ).perform(context)
    robot_description = {"robot_description": ParameterValue(value=robot_description_str, value_type=str)}

    controllers_file = PathJoinSubstitution([pkg_share, "config", "controllers.yaml"])

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[robot_description, {"use_sim_time": True}],
        ),
        # Runs MuJoCo and the controller manager in one process.
        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            emulate_tty=True,
            output="both",
            parameters=[{"use_sim_time": True}, ParameterFile(controllers_file)],
            on_exit=Shutdown(),
        ),
    ]

    for controller in CONTROLLERS:
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller, "--param-file", controllers_file],
                output="both",
            )
        )

    for arm in ("left", "right"):
        for kind, condition in (("joint_trajectory", IfCondition), ("forward_position", UnlessCondition)):
            nodes.append(Node(package="controller_manager", executable="spawner",
                              arguments=[f"{arm}_{kind}_controller", "--param-file", controllers_file],
                              condition=condition(LaunchConfiguration("trajectory_control")), output="both"))

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("trajectory_control", default_value="false",
                                  description="Expose FollowJointTrajectory actions for shared execution clients"),
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Run without the MuJoCo window",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
