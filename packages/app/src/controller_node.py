#!/usr/bin/env python3
"""A Node to control the duckiebot via the computer through keyboard or autopilote."""

import os
import sys

import cv2
import numpy as np

import rospy
from app.msg import WheelsSpeed
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int16
from utils.application import DuckieControlApp
from utils.autopilote import AutoPilote


class ControllerNode(object):
    """Controller Node for the duckiebot.

    The control is done through a pygame interface defined in DuckieControlApp.

    Subscribe to ~image/compressed to receive image from the camera.
    Publish to ~velocities to publish the Wheels velocities command.
    Publish to ~save_cmd to send save commands to writer node.
    """

    def __init__(self, node_name, resources_path):
        """Initialise the controller.

        Args:
            node_name: Default name of the node.
            resources_path: Path to the resource folder, containing model weights for autopilote.
        """
        # Init node
        rospy.init_node(node_name, anonymous=True)

        # Init pygame app
        self.app = DuckieControlApp()

        # Register publishers and subscriber
        self.pub_velocities = rospy.Publisher("~velocities", WheelsSpeed, queue_size=10)
        self.pub_save_cmd = rospy.Publisher("~save_cmd", Int16, queue_size=1)
        self.sub = rospy.Subscriber("~image/compressed", CompressedImage, self.callback)

        # Init Autopilote
        self.auto_pilote = AutoPilote(os.path.join(resources_path, "actor"))
        # Disable by default
        self.auto_pilote_mode = False

    def callback(self, data):
        """Callback on image receiving.

        Args:
            data: Image data. Should be of type CompressedImage.
        """
        # Load image from topic
        np_arr = np.fromstring(data.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)

        # Convert from BGR (used by cv2) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Add image to autopilote stack
        self.auto_pilote.add_to_stack(image)

        # Send data to the app, to display
        self.app.register_image(image)
        self.app.register_stack(self.auto_pilote.stack)

        # If autopilote is activated
        if self.auto_pilote_mode:
            # Get action from pilote
            action = self.auto_pilote.get_action()

            # Convert to WheelsSpeed msg and publish
            s = self.get_wheels_speed(action)
            self.pub_velocities.publish(s)

            # Send speed info to display
            self.app.register_action(action)

        # Render current image, stack and speed
        self.app.render()

    def run(self):
        """Run loop."""
        # 30 Hz
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            # Run a step on the app, to get the action from keyboard
            action, autopilote_toggle, save_cmd = self.app.step(self.auto_pilote_mode)

            # If an action is returned by app.step,
            # i.e. if self.auto_pilote_mode was True
            if action:
                s = self.get_wheels_speed(action)
                self.pub_velocities.publish(s)

            # Toggle autopilote if prompted by the user
            if autopilote_toggle:
                self.auto_pilote_mode = not self.auto_pilote_mode

            # Save a 500 timesteps episodes if prompted
            if save_cmd:
                self.pub_save_cmd.publish(500)

            # Sleep until next step
            rate.sleep()

    def get_wheels_speed(self, action):
        """Convert action list to WheelsSpeedCmd."""
        s = WheelsSpeed()
        s.left = action[0]
        s.right = action[1]
        return s


if __name__ == "__main__":
    # create the node
    node = ControllerNode("controller", sys.argv[1])

    node.run()

    # keep spinning
    rospy.spin()
