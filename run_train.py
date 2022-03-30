import torch
import random
import numpy as np

from train import learn_embedding


def set_seed(seed=225):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)


def set_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_config():
    return {'batch_size': 256, 'num_workers': 2, 'temperature': 0.5,
            'epochs': 50, 'lr': 3e-4, 'weight_decay': 1e-6,
            'device': set_device(), 'num_views': 2, 'pin_memory': True,
            'use_wandb': True, 'save_model': False, 'load_model': False}


def main():
    set_seed()
    learn_embedding(get_config())


if __name__ == "__main__":
    main()  # TODO: gradient accumulation
