#!/usr/bin/env python

import numpy as np
import pygame
import cv2
from my_package.msg import WheelsSpeed

IMG_W = 640
IMG_H = 480
STATE_W = 160
STATE_H = 120
GAP = 20

BACKGROUND = (41, 44, 48)

BAR_COLOR = (151, 152, 153)
BAR_X = (IMG_W + 2 * GAP) / 2
BAR_Y = 2 * GAP + IMG_H
BAR_LEN = IMG_W - 2 * GAP
BAR_H = 6

INDICATOR_COLOR = (255, 224, 84)
INDICATOR_H = 10
INDICATOR_WIDTH = 5

class Display:
    def __init__(self, screen):
        self.screen = screen

        self.speed = WheelsSpeed()
        self.stack = []
        self.image = np.zeros((IMG_W, IMG_H, 3))

        self.render_bar()

    def render(self):
        self.screen.fill(BACKGROUND)

        self.render_image()
        self.render_states()
        self.render_indicator()

        pygame.display.update()

    def render_image(self):
        surf = pygame.surfarray.make_surface(self.image)
        self.screen.blit(surf, (GAP, GAP))

    def render_states(self):
        for i, frame in enumerate(self.stack):
            # Resize
            s = cv2.resize(frame, (STATE_W, STATE_H))

            # Convert to 3d 0, 255
            s = ((s * 0.5) + 0.5) * 255
            s = s[..., np.newaxis].repeat(3, -1).astype("uint8")

            s = pygame.surfarray.make_surface(s)

            pos = (IMG_W + 2 * GAP, GAP + i * (IMG_H + GAP / 2))
            self.screen.blit(s, pos)

    def render_indicator(self):
        x = BAR_X + (self.speed.left - self.speed.right) * BAR_LEN / 2
        pygame.draw.line(self.screen, INDICATOR_COLOR, (x, BAR_Y - INDICATOR_H / 2), (BAR_X, BAR_Y + INDICATOR_H / 2))

    def render_bar(self):
        pygame.draw.line(self.screen, BAR_COLOR, (BAR_X - BAR_LEN / 2, BAR_Y), (BAR_X + BAR_LEN / 2, BAR_Y))
        pygame.draw.line(self.screen, BAR_COLOR, (BAR_X, BAR_Y - BAR_H / 2), (BAR_X, BAR_Y + BAR_H / 2))
