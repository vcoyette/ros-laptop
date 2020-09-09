FROM ros:melodic 

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python-catkin-tools \
 && rm -rf /var/lib/apt/lists/*

# Create Catkin workspace
ENV CATKIN_WS_DIR=/code/catkin_ws
WORKDIR ${CATKIN_WS_DIR}

# Upgrade pip
RUN pip3 install --upgrade pip

# install python dependencies
COPY ./dependencies-py.txt .
RUN pip3 install -r dependencies-py.txt

# copy the source code
COPY . ./src

# build packages
RUN . /opt/ros/melodic/setup.sh && \
  catkin build

# define command
CMD ["bash", "-c", "./src/launch.sh"]

ENV ROS_MASTER_IP 192.168.43.99
ENV ROS_MASTER_URI "http://$ROS_MASTER_IP:11311/"
ENV ROS_IP 192.168.43.228
