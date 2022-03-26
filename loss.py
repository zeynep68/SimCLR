import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, view1, view2):
        return torch.tensor(0)
