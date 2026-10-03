from setuptools import find_packages, setup

setup(
    name='pinocchio_teaching_examples', version='0.1.0', packages=find_packages(exclude=['test']),
    data_files=[('share/ament_index/resource_index/packages', ['resource/pinocchio_teaching_examples']),
                ('share/pinocchio_teaching_examples', ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='NU CERAS', maintainer_email='ceras@northwestern.edu',
    description='Runnable examples for introductory robotics labs.', license='Apache-2.0',
    entry_points={'console_scripts': ['model_info = pinocchio_teaching_examples.demos:model_info', 'forward_kinematics = pinocchio_teaching_examples.demos:forward_kinematics', 'jacobian = pinocchio_teaching_examples.demos:jacobian', 'gravity = pinocchio_teaching_examples.demos:gravity', 'inverse_kinematics = pinocchio_teaching_examples.demos:inverse_kinematics', 'hand_position = pinocchio_teaching_examples.hand_position:main']},
)
