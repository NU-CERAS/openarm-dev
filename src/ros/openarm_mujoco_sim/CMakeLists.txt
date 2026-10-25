cmake_minimum_required(VERSION 3.8)
project(openarm_mujoco_sim)
find_package(ament_cmake REQUIRED)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
# Keep one canonical copy of the robot assets in this repository.
install(DIRECTORY ../../../robots/v1.0/models/ DESTINATION share/${PROJECT_NAME}/mjcf)
install(DIRECTORY ../../../robots/v1.0/urdf DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/simple_motion.py DESTINATION lib/${PROJECT_NAME})
ament_package()
