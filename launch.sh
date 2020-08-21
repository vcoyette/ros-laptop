#!/bin/bash

set -e

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------
source /code/catkin_ws/devel/setup.bash
roslaunch app multiple_nodes.launch veh:=mobyduck
