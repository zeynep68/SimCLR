import torch
import random
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from augmentation import DataAugmentation
from torchvision.datasets import STL10

from simclr import SimCLRNet
from loss import ContrastiveLoss


def set_seed(seed=225):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)


def set_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_unlabeled_data(batch_size, num_workers):
    trainset = STL10(root='./data', download=True, transform=DataAugmentation(),
                     split='unlabeled')
    return get_dataloader(trainset, batch_size, num_workers)


def get_dataloader(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers)


def train_one_epoch(config, trainloader, model, optimizer, criterion):
    for (view1, view2), _ in trainloader:  # num_batches
        view1 = view1.to(config['device'])
        view2 = view2.to(config['device'])

        train_step(model, optimizer, criterion, view1, view2)


def train_step(model, optimizer, criterion, view1, view2):
    optimizer.zero_grad()

    outputs1 = model(view1)
    outputs2 = model(view2)

    loss = criterion(outputs1, outputs2)
    loss.backward()
    optimizer.step()


def get_config():
    return {'batch_size': 64, 'num_workers': 1, 'T': 1., 'epochs': 1, 'lr': 3e-4,
            'weight_decay': 10e-6, 'device': set_device()}


def pretrain(config):
    # TODO: is this correct??
    # or is the order in which batches are created same??
    trainloader = load_unlabeled_data(256, config['num_workers'])

    model = SimCLRNet()
    criterion = ContrastiveLoss()
    optimizer = Adam(model.parameters())

    for _ in range(config['epochs']):
        train_one_epoch(config, trainloader, model, criterion, optimizer)


if __name__ == "__main__":
    set_seed()
    pretrain(get_config())
