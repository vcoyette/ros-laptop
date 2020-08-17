# Dockerfile for ros-laptop

# Base image
ARG OS=ubuntu
ARG DISTRO=focal
ARG ARCH=amd64

FROM ${ARCH}/${OS}:${DISTRO}

RUN apt-get update && apt-get install -y \
    build-essential

# Install ROS (http://wiki.ros.org/noetic/Installation/Ubuntu)
RUN apt-key adv \
    --keyserver hkp://keyserver.ubuntu.com:80 \
    --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list

RUN apt-get update && apt-get install -y \
    ros-noetic-rospy \
    ros-noetic-robot \
    ros-noetic-cv-bridge


# Create and set workspace
WORKDIR /workspace

