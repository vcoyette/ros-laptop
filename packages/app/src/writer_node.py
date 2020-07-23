#!/usr/bin/env python
"""A Node to write images to disk."""

import os
import rospy
import cv2
import numpy as np
from std_msgs.msg import Int16
from sensor_msgs.msg import CompressedImage
import PIL.Image as Image

EPISODE_REGEXP = "episode_{:02}"
IMG_TEMPLATE = "image_{:04}.png"
ROOT_FOLDER = os.path.join("/transfer", "duckie")


class WriterNode(object):
    """Writer node to save images from the camera.

    Subscribe to ~save_cmd to receive the command to save an episode.
    Subscribe to ~image/compressed when the saving of an episode is being performed.

    !!Requires Docker to be ran mounting /transfer folder with --volume option.
    """

    def __init__(self, node_name):
        """Initialise the controller.

        Args:
            node_name: Default name of the node.
        """
        # Init node
        rospy.init_node(node_name, anonymous=True)

        # Register subscriber
        self.save_sub = rospy.Subscriber("~save_cmd", Int16, self.save_episode)

        # Init episode counter
        self.episode = 0
        self.folder = os.path.join(ROOT_FOLDER, EPISODE_REGEXP.format(self.episode))

        # Init image list
        self.images = []

    def save_episode(self, data):
        """Callback on save cmd receive.

        Args:
            data: the length of the episode to save. Don't use too long episodes,
                  as images are stored in memory and written to disk only at the
                  end of episode. This is to prevent image loss because writing
                  on disk may be long.
        """
        self.episode_length = data.data

        # Create an unexisting folder for episode
        while os.path.exists(self.folder):
            self.episode += 1
            self.folder = os.path.join(ROOT_FOLDER, EPISODE_REGEXP.format(self.episode))

        os.makedirs(self.folder)

        # Register a subscriber to image topic
        self.image_sub = rospy.Subscriber(
            "~image/compressed", CompressedImage, self.register_images
        )

    def register_images(self, data):
        """Callback on image receiving.

        Args:
            data: compressed image from the camera.
        """
        print("Saving image...", len(self.images))

        # If episode is finished
        if len(self.images) >= self.episode_length:
            # Unsubscribes from ~image/compressed
            self.image_sub.unregister()

            # Actually write images on disk
            self.save_images()

            # Update episode counter
            self.episode += 1

        # Load image from topic
        np_arr = np.fromstring(data.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)

        # Convert from BGR (used by cv2) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Add to image list
        self.images.append(image)

    def save_images(self):
        """Save images on disk."""
        print("Writing images on disk...")
        for i, image in enumerate(self.images):
            # Create filename
            filename = IMG_TEMPLATE.format(i)

            # Convert to PIL img
            img = Image.fromarray(image)

            # Save
            img.save(os.path.join(self.folder, filename))

        print("Done saving episode")
        # Reinit images list
        self.images = []


if __name__ == "__main__":
    # create the node
    node = WriterNode("writer")

    # keep spinning
    rospy.spin()
