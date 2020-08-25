"""Define Feature Extractor."""

import torch
import torch.nn as nn

from utils.whitening import WTransform2d


class FeatureExtractor(nn.Module):
    """Feature Extractor."""

    def __init__(self, g=4):
        """Initialize actor."""
        super(FeatureExtractor, self).__init__()

        self.lr = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=2)
        self.dwts1 = WTransform2d(32, g)
        self.dwtt1 = WTransform2d(32, g)
        self.dwtt1_aug = WTransform2d(32, g)
        self.dwt1 = [self.dwts1, self.dwtt1, self.dwtt1_aug]
        self.gamma1 = nn.Parameter(torch.ones(32, 1, 1))
        self.beta1 = nn.Parameter(torch.zeros(32, 1, 1))

        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.dwts2 = WTransform2d(32, g)
        self.dwtt2 = WTransform2d(32, g)
        self.dwtt2_aug = WTransform2d(32, g)
        self.dwt2 = [self.dwts2, self.dwtt2, self.dwtt2_aug]
        self.gamma2 = nn.Parameter(torch.ones(32, 1, 1))
        self.beta2 = nn.Parameter(torch.zeros(32, 1, 1))

        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.dwts3 = WTransform2d(32, g)
        self.dwtt3 = WTransform2d(32, g)
        self.dwtt3_aug = WTransform2d(32, g)
        self.dwt3 = [self.dwts3, self.dwtt3, self.dwtt3_aug]
        self.gamma3 = nn.Parameter(torch.ones(32, 1, 1))
        self.beta3 = nn.Parameter(torch.zeros(32, 1, 1))

        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)
        self.dwts4 = WTransform2d(32, g)
        self.dwtt4 = WTransform2d(32, g)
        self.dwtt4_aug = WTransform2d(32, g)
        self.dwt4 = [self.dwts4, self.dwtt4, self.dwtt4_aug]
        self.gamma4 = nn.Parameter(torch.ones(32, 1, 1))
        self.beta4 = nn.Parameter(torch.zeros(32, 1, 1))

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, target=0):
        # CNN
        x = self.conv1(x)
        x = self.dwt1[target](x) * self.gamma1 + self.beta1
        x = self.lr(x)

        x = self.conv2(x)
        x = self.dwt2[target](x) * self.gamma2 + self.beta2
        x = self.lr(x)

        x = self.conv3(x)
        x = self.dwt3[target](x) * self.gamma3 + self.beta3
        x = self.lr(x)

        x = self.conv4(x)
        x = self.dwt4[target](x) * self.gamma4 + self.beta4
        x = self.lr(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)

        return x
