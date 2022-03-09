import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from augmentation import DataAugmentation
from torchvision.datasets import STL10


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


def train(config):
    train = load_unlabeled_data(256, config['num_workers'])

    for (view1, view2), _ in train:
        view1 = view1.to(config['device'])
        view2 = view2.to(config['device'])


def config():
    return {'batch_size': 64, 'num_workers': 1, 'T': 1., 'epochs': 100,
            'lr': 4.8, 'weight_decay': 10e-6, 'device': set_device()}


if __name__ == "__main__":
    train(config())
