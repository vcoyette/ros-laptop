#!/usr/bin/env python
"""Autopilote, using Imitation learning."""

from utils.model import Actor
import torch
from collections import deque
import cv2


class AutoPilote:
    """Class defining the auto pilote behaviour."""

    def __init__(self, weights_path, num_stack=4, shape=(60, 80)):
        """Initialise AutoPilote.

        [TODO:description]

        Args:
            weights_path: Path to the network weights file.
            num_stack (int, optional): Number of images to stack in a state.
            shape (Tuple[int], optional): Shape of an image in the stack.
        """
        self._actor = Actor(2, 1)
        self._actor.load_state_dict(
            torch.load(weights_path, map_location=torch.device("cpu"))
        )
        self._stack = deque(maxlen=num_stack)
        self._num_stack = num_stack
        self._shape = shape

    def get_action(self):
        """Return the action to perform based on current stack."""
        state = torch.FloatTensor(self._stack)
        state = state.unsqueeze(0)
        return self._actor(state).detach().numpy().flatten()

    def add_to_stack(self, image):
        """Add an image to the state stack."""
        # Downscale images
        obs = cv2.resize(image, self._shape, interpolation=cv2.INTER_AREA)
        # Transform to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # Scale to 0:1
        obs = (obs - float(obs.min())) / (obs.max() - obs.min())
        # Scale to -1:1
        obs = (obs - 0.5) / 0.5

        # If the stack is empty (namely 1st timestep)
        if not self._stack:
            self._fill_stack(obs)

        # Add observation to stack
        self._stack.append(obs)

    def _fill_stack(self, image):
        """Fill the stack with image."""
        for _ in range(self._num_stack):
            self._stack.append(image)
