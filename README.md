# ROS Laptop

This repository contains the code to build a docker image to run on a computer to control the duckiebot. It is supposed to be used with https://github.com/vcoyette/ros-duckie running on the duckie.

The docker image doesn't contain a ros master but connects to the one running on the duckiebot. It contains one actor node whichs subscribes to the images published by the camera and publish the wheels command computed by the Deep model.

Usage:
1. Build:
```bash
make build
```
2. Run:
```bash
make run
```

