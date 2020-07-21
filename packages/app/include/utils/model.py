"""Define actor architecture."""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Actor network."""

    def __init__(self, action_dim, max_action):
        """Initialize actor.

        Keyword Arguments:
        action_dim -- Dimension of the action space
        max_action -- Max value of an action
        """
        super(Actor, self).__init__()
        flat_size = 32 * 2 * 4

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 512)
        self.lin2 = nn.Linear(512, action_dim)

        self.max_action = max_action

    def forward(self, x):
        """Forward pass x."""
        # CNN
        x = self.lr(self.conv1(x))
        x = self.lr(self.conv2(x))
        x = self.lr(self.conv3(x))
        x = self.lr(self.conv4(x))

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)

        # Output
        x = self.lr(self.lin1(x))
        x = self.max_action * self.sigm(self.lin2(x))

        return x
