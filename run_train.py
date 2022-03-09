from torch.utils.data import DataLoader
from augmentation import DataAugmentation
from torchvision.datasets import STL10


def load_unlabeled_data(batch_size, num_workers):
    transform = DataAugmentation()
    trainset = STL10(root='./data', download=True, transform=transform,
                     split='unlabeled')
    return dataloader(trainset, batch_size, num_workers)


def dataloader(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers)


def train(config):
    train = load_unlabeled_data(100_000, config['num_workers'])

    for data in train:
        inputs, labels = data


def config():
    return {'batch_size': 64, 'num_workers': 1, 'T': 1., 'epochs': 100,
            'lr': 4.8, 'weight_decay': 10e-6}


if __name__ == "__main__":
    train(config())
