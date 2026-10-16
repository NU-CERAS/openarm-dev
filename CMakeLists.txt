cmake_minimum_required(VERSION 3.8)
project(openarm_mujoco_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch mjcf urdf
  DESTINATION share/${PROJECT_NAME}
)

# `ros2 run openarm_mujoco_sim simple_motion.py`
install(PROGRAMS scripts/simple_motion.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
