"""Define the classifier Architecture."""

import torch
import torch.nn as nn


class Classifier(nn.Module):
    """Classifier network."""

    def __init__(self, action_dim, max_action):
        nn.Module.__init__(self)

        flat_size = 32 * 2 * 4

        self.lr = nn.LeakyReLU()
        self.sigm = nn.Sigmoid()

        self.lin1 = nn.Linear(flat_size, 512)
        self.bns1 = nn.BatchNorm1d(512, affine=False)
        self.bnt1 = nn.BatchNorm1d(512, affine=False)
        self.bnt1_aug = nn.BatchNorm1d(512, affine=False)
        self.bn1 = [self.bns1, self.bnt1, self.bnt1_aug]
        self.gamma1 = nn.Parameter(torch.ones(1, 512))
        self.beta1 = nn.Parameter(torch.zeros(1, 512))

        self.lin2 = nn.Linear(512, action_dim)
        self.bns2 = nn.BatchNorm1d(action_dim, affine=False)
        self.bnt2 = nn.BatchNorm1d(action_dim, affine=False)
        self.bnt2_aug = nn.BatchNorm1d(action_dim, affine=False)
        self.bn2 = [self.bns2, self.bnt2, self.bnt2_aug]
        self.gamma2 = nn.Parameter(torch.ones(1, action_dim))
        self.beta2 = nn.Parameter(torch.zeros(1, action_dim))

        self.max_action = max_action

    def forward(self, x, target=0):
        x = self.lin1(x)
        x = self.bn1[target](x) * self.gamma1 + self.beta1
        x = self.gamma1 * x + self.beta1
        x = self.lr(x)

        x = self.lin2(x)
        x = self.bn2[target](x) * self.gamma2 + self.beta2
        x = self.gamma2 * x + self.beta2

        return self.max_action * self.sigm(x)
