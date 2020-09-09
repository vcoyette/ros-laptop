#!/usr/bin/env python
"""UI definition for the application."""

import cv2
import numpy as np

import pygame
from app.msg import WheelsSpeed

# Screen definition
SCREEN_W = 860
SCREEN_H = 540

# GAP Size
GAP = 20

# Image size
IMG_W = 640
IMG_H = 480

# State stack images size
STATE_W = 160
STATE_H = 120
# Gap between images in stack
STATE_GAP = 10

# Background color
BACKGROUND = (41, 44, 48)

# Bar color
BAR_COLOR = (151, 152, 153)
# Bar center Coordinates
BAR_X = (IMG_W + 2 * GAP) / 2
BAR_Y = IMG_H + 2 * GAP
# Lenght of the bar
BAR_LEN = IMG_W - 2 * GAP
# Height of the bar (vertical bar in the middle of the bar)
BAR_H = 6

# Speed indicator color and size
INDICATOR_COLOR = (255, 224, 84)
INDICATOR_H = 10
INDICATOR_WIDTH = 5


class Display(object):
    """Class to display interface.

    Contains three informations:
        1. The image from the camera.
        2. The stack of grayscale images in agent state.
        3. A bar on which an indicator shows if the robot turns left or right.
    """

    def __init__(self):
        """Initialise display."""
        # Init screen
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))

        # Init informations to display
        self._action = [0, 0]
        self._stack = []
        self._image = np.zeros((IMG_H, IMG_W, 3))

    def render(self):
        """Render current display."""
        # Set background color
        self.screen.fill(BACKGROUND)

        # Render component
        self._render_image()
        self._render_state()
        self._render_bar()
        self._render_indicator()

        # Update screen with the buffer
        pygame.display.update()

    def _render_image(self):
        """Render the image from camera."""
        # Swap axes as cv2 image are height, width indexed, pygame width, height
        image = self._image.swapaxes(0, 1)

        # Make a surface from the array
        surf = pygame.surfarray.make_surface(image)

        # Display the array on screen
        self.screen.blit(surf, (GAP, GAP))

    def _render_state(self):
        """Render the state."""
        for i, frame in enumerate(self._stack):
            # Resize
            s = cv2.resize(frame, (STATE_W, STATE_H))

            # Convert to 0, 255
            s = ((s * 0.5) + 0.5) * 255
            # Convert to 3d
            s = s[..., np.newaxis].repeat(3, -1).astype("uint8")

            # Swap axis again (h, w) -> (w, h)
            s = s.swapaxes(0, 1)

            # Make a surface from the array
            s = pygame.surfarray.make_surface(s)

            # Get position of current image
            pos = (IMG_W + 2 * GAP, GAP + i * (STATE_H + STATE_GAP))

            # Display the array on screen
            self.screen.blit(s, pos)

    def _render_indicator(self):
        """Render the indicator on the bar."""
        # The indicator differ from the center of the bar by the difference
        # between left and right speeds
        x = BAR_X + (self._action[0] - self._action[1]) * BAR_LEN / 2

        # Draw the indicator
        pygame.draw.line(
            self.screen,
            INDICATOR_COLOR,
            (x, BAR_Y - INDICATOR_H / 2),
            (x, BAR_Y + INDICATOR_H / 2),
            INDICATOR_WIDTH,
        )

    def _render_bar(self):
        """Render the bar."""
        # Horizontal bar
        pygame.draw.line(
            self.screen,
            BAR_COLOR,
            (BAR_X - BAR_LEN / 2, BAR_Y),
            (BAR_X + BAR_LEN / 2, BAR_Y),
        )
        # Vertical line at the center
        pygame.draw.line(
            self.screen,
            BAR_COLOR,
            (BAR_X, BAR_Y - BAR_H / 2),
            (BAR_X, BAR_Y + BAR_H / 2),
        )

    @property
    def action(self):
        """The last action of the duckie."""
        return self._action

    @action.setter
    def action(self, action):
        self._action = action

    @property
    def stack(self):
        """The last state stack."""
        return self._stack

    @stack.setter
    def stack(self, stack):
        self._stack = stack

    @property
    def image(self):
        """The last image captured by the camera."""
        return self._image

    @image.setter
    def image(self, image):
        self._image = image
