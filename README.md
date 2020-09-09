# ROS Laptop

This repository contains the code to build a docker image to run on a computer to control the duckiebot. It is supposed to be used with https://github.com/vcoyette/ros-duckie running on the duckie.

The docker image doesn't contain a ros master but connects to the one running on the duckiebot. It contains one actor node whichs subscribes to the images published by the camera and publish the wheels command computed by the Deep model.

## Requirements
This app requires docker to be built and ran. 

It also requires to know the ip of the duckie.
To find it, you can run:
```bash 
ping mobyduck.local
```
(replace mopbyduck with the hostname of your duckie).

Store this ip, we will refer to it as DUCKIE_IP later.

## Usage

1. Build:
```bash
make build
```

2. Run:
Before running, ensure a ROS master is running on the duckie (run [ros-duckie](https://github.com/vcoyette/ros-duckie)).

```bash
make run duckie_ip=DUCKIE_IP
```
Replace DUCKIE_IP by the IP of your duckie.

## Troubleshooting
No available video device:
You need to allow docker to access your X server. You can run:

```bash
xhost +"local:docker@"
```

## UML

Here is a sequence diagram presenting each ros node and topics.
The infinite loop is launched on startup. If the user press enter, the episode saving loop is launched for 500 timesteps. The only difference is the 

``` plantuml
== Infinite Loop ==
camera_node -> controller_node: image/compressed
controller_node -> motor_node: velocities  
 
== Episode saving ==
controller_node -> writer_node: save_cmd
camera_node -> controller_node: image/compressed
camera_node -> writer_node: image/compressed
controller_node -> motor_node: velocities  
```
