#!/usr/bin/env python

from my_package.model import Actor
import torch
from collections import deque
import cv2


class AutoPilote:
    def __init__(self, weights_path, num_stack=4, shape=(60, 80)):
        self.actor = Actor(2, 1)
        self.actor.load_state_dict(torch.load(weights_path))
        self.stack = deque(maxlen=num_stack)
        self.num_stack = num_stack
        self.shape = shape

    def get_action(self):
        state = torch.FloatTensor(self.stack)
        state = state.unsqueeze(0)
        return self.actor(state).detach().numpy().flatten()

    def add_to_stack(self, image):
        obs = cv2.resize(image, self.shape)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = obs / 255.0
        obs = (obs - 0.5) / 0.5
        if len(self.stack) == 0:
            self.fill_stack(obs)
        self.stack.append(obs)

    def fill_stack(self, image):
        for _ in range(self.num_stack):
            self.stack.append(image)
