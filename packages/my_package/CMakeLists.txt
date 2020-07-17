cmake_minimum_required(VERSION 2.8.3)
project(my_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  WheelsSpeed.msg
)

generate_messages(
    DEPENDENCIES
)

catkin_package(CATKIN_DEPENDS message_runtime )
