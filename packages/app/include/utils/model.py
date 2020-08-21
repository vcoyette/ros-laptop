"""Define actor architecture."""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
defaults = {"use_bNorm": False}


class Actor(nn.Module):
    """Actor network."""

    def __init__(self, action_dim, max_action, **params):
        """Initialize actor.
        Keyword Arguments:
        action_dim -- Dimension of the action space
        max_action -- Max value of an action
        """
        super(Actor, self).__init__()
        flat_size = 32 * 2 * 4
        self.use_bNorm = True

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        if self.use_bNorm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(32)
            self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(0.5)

        self.lin1 = nn.Linear(flat_size, 512)
        self.lin2 = nn.Linear(512, action_dim)

        self.max_action = max_action

    def forward(self, x):
        """Forward pass x."""
        # CNN
        if self.use_bNorm:
            x = self.bn1(self.lr(self.conv1(x)))
            x = self.bn2(self.lr(self.conv2(x)))
            x = self.bn3(self.lr(self.conv3(x)))
            x = self.bn4(self.lr(self.conv4(x)))
        else:
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
