import torch.nn as nn
from torchvision.models import resnet18


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18()
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(self, resnet_dim, embedding_dim):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(resnet_dim, resnet_dim),
                                  nn.ReLU(),
                                  nn.Linear(resnet_dim, embedding_dim))

    def forward(self, x):
        return self.head(x)


class SimCLRNet(nn.Module):
    def __init__(self, embedding_dim=128, resnet_dim=512):
        super().__init__()
        self.encoder = Encoder()
        self.head = ProjectionHead(resnet_dim, embedding_dim)

    def forward(self, x):
        return self.encoder(x)

    def project(self, x):
        x = self.forward(x)
        return self.head(x)
