from setuptools import find_packages, setup

setup(
    name='ros_teaching_examples', version='0.1.0', packages=find_packages(exclude=['test']),
    data_files=[('share/ament_index/resource_index/packages', ['resource/ros_teaching_examples']),
                ('share/ros_teaching_examples', ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='NU CERAS', maintainer_email='ceras@northwestern.edu',
    description='Runnable examples for introductory robotics labs.', license='Apache-2.0',
    entry_points={'console_scripts': ['angle_publisher = ros_teaching_examples.angle_publisher:main', 'angle_subscriber = ros_teaching_examples.angle_subscriber:main', 'joint_motion = ros_teaching_examples.joint_motion:main']},
)
