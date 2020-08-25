"""Integrated DANN network."""

import torch.nn as nn


class DWTModel(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(DWTModel, self).__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x, target=0):
        x = self.feature_extractor(x, target)
        return self.classifier(x, target)
