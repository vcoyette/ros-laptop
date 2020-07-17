#!/usr/bin/env python

import os
import rospy
from duckietown import DTROS
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
from my_package.msg import WheelsSpeed
import pygame
from my_package.display import Display
from my_package.autopilote import AutoPilote
import sys


class MyNode(DTROS):
    def __init__(self, node_name, resources_path):
        pygame.init()
        screen = pygame.display.set_mode((860, 540))

        self.display = Display(screen)
        # initialize the DTROS parent class
        super(MyNode, self).__init__(node_name=node_name)
        # construct publisher
        self.pub = rospy.Publisher("~velocities", WheelsSpeed, queue_size=10)
        self.sub = rospy.Subscriber("~image/compressed", CompressedImage, self.callback)

        self.auto_pilote_mode = False
        self.auto_pilote = AutoPilote(os.path.join(resources_path, "actor"))


    def callback(self, data):
        np_arr = np.fromstring(data.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.auto_pilote.add_to_stack(image)

        self.display.image = image
        self.display.stack = self.auto_pilote.stack

        if self.auto_pilote_mode:
            s = WheelsSpeed()
            a = self.auto_pilote.get_action()
            s.left = float(a[0])
            s.right = float(a[1])
            self.pub.publish(s)
            self.display.speed = s

        self.display.render()


    def run(self):
        rate = rospy.Rate(30)  # 1Hz

        while not rospy.is_shutdown():
            pygame.event.pump()

            s = WheelsSpeed()
            keys = pygame.key.get_pressed()

            if keys[pygame.K_a]:
                self.auto_pilote_mode = not self.auto_pilote_mode

            if not self.auto_pilote_mode:
                if keys[pygame.K_UP]:
                    s.left = 0.44
                    s.right = 0.44
                if keys[pygame.K_DOWN]:
                    s.left = -0.44
                    s.right = -0.44
                if keys[pygame.K_LEFT]:
                    s.left = 0.34
                    s.right = 0.46
                if keys[pygame.K_RIGHT]:
                    s.left = 0.46
                    s.right = 0.34
                self.pub.publish(s)
                self.display.speed = s
            rate.sleep()


if __name__ == "__main__":
    # create the node
    node = MyNode("actor", sys.argv[1])

    node.run()

    # keep spinning
    rospy.spin()
