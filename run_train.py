import torch
import random
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import STL10

from simclr import SimCLRNet
from logs import initialize_logging, log_loss
from loss import ContrastiveLoss
from augmentation import DataAugmentation


def set_seed(seed=225):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)


def set_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_unlabeled_data(config):
    trainset = STL10(root='./data', download=True, transform=DataAugmentation(),
                     split='unlabeled')
    return get_dataloader(trainset, config['batch_size'], config['num_workers'],
                          config['pin_memory'])


def get_dataloader(dataset, batch_size, num_workers, pin_memory):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory)


def train_one_epoch(config, trainloader, model, optimizer, criterion):
    model.to(config['device'])
    model.train()

    for (view1, view2), _ in trainloader:  # num_batches
        view1 = view1.to(config['device'])
        view2 = view2.to(config['device'])

        criterion.batch_size = view1.shape[0]  # last batch has less images

        train_step(model, optimizer, criterion, view1, view2, config)


def train_step(model, optimizer, criterion, view1, view2, config):
    optimizer.zero_grad()

    outputs1 = model.project(view1)
    outputs2 = model.project(view2)

    loss = criterion(outputs1, outputs2)
    if config['use_wandb']:
        log_loss(loss.item())

    loss.backward()
    optimizer.step()


def get_config():
    return {'batch_size': 256, 'num_workers': 2, 'temperature': 0.5,
            'epochs': 5, 'lr': 3e-4, 'weight_decay': 1e-6,
            'device': set_device(), 'num_views': 2, 'pin_memory': True,
            'use_wandb': True}


def main(config):
    trainloader = load_unlabeled_data(config)

    model = SimCLRNet()
    criterion = ContrastiveLoss(config['batch_size'], config['device'])
    optimizer = Adam(model.parameters(), lr=config['lr'],
                     weight_decay=config['weight_decay'])

    if config['use_wandb']:
        initialize_logging()

    for e in range(config['epochs']):
        train_one_epoch(config, trainloader, model, optimizer, criterion)

    # TODO: save model


if __name__ == "__main__":
    set_seed()

    main(get_config())  # TODO: gradient accumulation

    # fine-tune on supervised stl10
