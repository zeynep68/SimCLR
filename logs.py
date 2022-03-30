import wandb

PREFIX = ''


def initialize_logging():
    """ Initialize logging with wandb. """
    wandb.init(project='SimCLR')
    wandb.run.name = PREFIX + wandb.run.name


def log_loss(loss):
    wandb.log(loss)
