"""Launch one robot with either manual sliders or the teaching motion node."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory('teaching_arm_description'))
    use_gui = LaunchConfiguration('use_gui')
    description = ParameterValue(Command([
        FindExecutable(name='xacro'), ' ', str(share / 'urdf/two_link_arm.urdf.xacro')
    ]), value_type=str)
    return LaunchDescription([
        DeclareLaunchArgument('use_gui', default_value='true',
                              description='Use sliders; false uses the motion example.'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': description}]),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
             condition=IfCondition(use_gui)),
        Node(package='ros_teaching_examples', executable='joint_motion',
             condition=UnlessCondition(use_gui)),
        Node(package='rviz2', executable='rviz2',
             arguments=['-d', str(share / 'rviz/teaching_arm.rviz')]),
    ])
