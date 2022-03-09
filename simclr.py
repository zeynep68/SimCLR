import torch.nn as nn
from torchvision.models import resnet50


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet50()
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(in_dim, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(x)


class SimCLRNet(nn.Module):
    def __init__(self, num_classes=10, pretrain=True):
        super().__init__()
        self.pretrain = pretrain

        self.encoder = Encoder()
        self.head = ProjectionHead(0, num_classes)

    def forward(self, x):
        x = self.encoder(x)

        # if prediction (classification) use h instead of g(h)
        return self.head(x) if self.pretrain else x
