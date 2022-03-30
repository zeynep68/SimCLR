import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import STL10

from simclr import SimCLRNet
from logs import initialize_logging, log_loss
from loss import ContrastiveLoss
from augmentation import DataAugmentation

PATH = './SimCLR.ckpt'


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


def learn_embedding(config):
    trainloader = load_unlabeled_data(config)

    model = SimCLRNet()
    criterion = ContrastiveLoss(config['batch_size'], config['device'])
    optimizer = AdamW(model.parameters(), lr=config['lr'],
                      weight_decay=config['weight_decay'])

    if config['use_wandb']:
        initialize_logging()

    for e in range(config['epochs']):
        train_one_epoch(config, trainloader, model, optimizer, criterion)

        # TODO: save model
        if config['save_model']:
            torch.save(model.state_dict(), PATH)
        print('test')
        if config['load_model']:
            model.load_state_dict(torch.load(PATH))
        print('test2')
