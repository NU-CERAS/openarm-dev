FROM ros:jazzy-ros-base-noble
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-rviz2 ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-robot-state-publisher ros-jazzy-xacro ros-jazzy-pinocchio \
    ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-moveit \
    python3-colcon-common-extensions python3-vcstool python3-rosdep \
    build-essential cmake git sudo liburdfdom-tools iproute2 kmod can-utils \
    xvfb x11-utils x11vnc novnc websockify openbox xterm mesa-utils \
    && rm -rf /var/lib/apt/lists/*
RUN echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu
COPY .devcontainer/desktop.sh /usr/local/bin/openarm-desktop
COPY .devcontainer/ros-shell.sh /etc/openarm-shell.sh
RUN chmod +x /usr/local/bin/openarm-desktop \
    && echo 'source /etc/openarm-shell.sh' >> /home/ubuntu/.bashrc
USER ubuntu
ENV DISPLAY=:1 LIBGL_ALWAYS_SOFTWARE=1 QT_X11_NO_MITSHM=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
WORKDIR /workspaces/openarm-dev
