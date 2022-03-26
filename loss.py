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

        mask = torch.eye(self.batch_size)
        mask = mask.repeat(self.num_views, self.num_views)

        # mask-out self contrast cases
        size = torch.arange(self.num_views * self.batch_size)
        mask[size, size] = 0

        mask = mask.to(self.device)
        print('mask:', mask.shape)
        print('cosine_sim:', cosine_similarity.shape)
        # just positive pairs
        logits = cosine_similarity * mask
        print('logits:', logits.shape)

        # compute log
        denominator = torch.exp(logits)
        denominator = torch.log(denominator)

        loss = logits - denominator

        return
