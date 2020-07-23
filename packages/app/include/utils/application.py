#!/usr/bin/env python
"""Pygame app to control the duckiebot."""

import pygame
from utils.display import Display

INSTRUCTIONS = """Keys:
    Arrows: Move the duckiebot if autopilote is not active.
    A: toggle autopilote mode.
"""


class DuckieControlApp(object):
    """Application class.

    This application contains an event listener to catch keyboard keys press
    and a UI to display information. All UI logic is defined in utils.display.
    """

    def __init__(self):
        """Initialise app."""
        # Init pygame
        pygame.init()

        # Init display
        self.display = Display()

        # Display commands
        print(INSTRUCTIONS)

    def register_image(self, image):
        """Register an image for future display."""
        self.display.image = image

    def register_stack(self, stack):
        """Register a state stack for future display."""
        self.display.stack = stack

    def register_action(self, action):
        """Register an action for future display."""
        self.display.action = action

    def step(self, auto_pilote_mode):
        """Run a step of app logic.

        This will check if an arrow key is pressed if in autopilote_mode and
        return the action to take, listen to the a key pressed and return
        toggle autopilote boolean if it is.
        """
        # Get keys currently being pressed
        keys = pygame.key.get_pressed()

        action = []

        # If in auto pilote mode, convert key to action
        if not auto_pilote_mode:
            # Default if no key is pressed
            action = [0, 0]
            if keys[pygame.K_UP]:
                action[0] = 0.44
                action[1] = 0.44
            if keys[pygame.K_DOWN]:
                action[0] = -0.44
                action[1] = -0.44
            if keys[pygame.K_LEFT]:
                action[0] = 0.34
                action[1] = 0.46
            if keys[pygame.K_RIGHT]:
                action[0] = 0.46
                action[1] = 0.34

            # Register action to display
            self.register_action(action)

        # Retrive pygame event list
        pygame.event.pump()

        autopilote_toggle = False
        save_cmd = False

        # If A have been pressed once, toggle autopilote
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                autopilote_toggle = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                save_cmd = True

        return action, autopilote_toggle, save_cmd

    def render(self):
        """Render display."""
        self.display.render()
