import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device, temperature=0.1, num_views=2):
        super().__init__()
        self.batch_size = batch_size
        self.num_views = num_views
        self.device = device

        self.temperature = temperature

    def forward(self, view1, view2):
        # l2 normalization
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)

        # concatenate view vectors along batch dim
        views = torch.cat((view1, view2), dim=0)

        cosine_similarity = (views @ views.T) / self.temperature

        # negative pairs are masked-out
        mask = torch.eye(self.batch_size)
        mask = mask.repeat(self.num_views, self.num_views)
        # mask-out self contrast cases
        size = torch.arange(self.num_views * self.batch_size)
        mask[size, size] = 0

        num_pos_pairs = mask.sum(axis=1)  # self.views * self.batch_size
        num_pos_pairs = num_pos_pairs.to(self.device)

        mask = mask.to(self.device)

        # just positive pairs - in each row one value for pos. pair
        nominator = cosine_similarity * mask
        nominator = nominator.sum(axis=1)

        # for denominator just mask-out self contrast cases
        # contains negative & positive pairs
        mask = torch.ones_like(mask)
        mask[size, size] = 0

        mask = mask.to(self.device)

        # compute log
        denominator = torch.exp(cosine_similarity) * mask
        denominator = denominator.sum(axis=1)
        denominator = torch.log(denominator)

        loss = nominator - denominator  # - because of log
        loss = - (nominator / num_pos_pairs)

        return loss.mean()
