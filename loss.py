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

    def mask_out_self_contrast_cases(self, mask):
        size = torch.arange(self.num_views * self.batch_size)
        mask[size, size] = 0

        return mask

    def get_positive_pairs(self):
        mask = torch.eye(self.batch_size)
        mask = mask.repeat(self.num_views, self.num_views)
        mask = self.mask_out_self_contrast_cases(mask)

        return mask

    def forward(self, view1, view2):
        # l2 normalization
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)

        # concatenate view vectors along batch dim
        views = torch.cat((view1, view2), dim=0)

        cosine_similarity = (views @ views.T) / self.temperature

        mask = self.get_positive_pairs()
        mask = mask.to(self.device)

        # just positive pairs - in each row one value for pos. pair
        nominator = cosine_similarity * mask
        nominator = nominator.sum(axis=1)

        # for denominator just mask-out self contrast cases
        # contains negative & positive pairs
        mask = torch.ones_like(mask)
        mask = self.mask_out_self_contrast_cases(mask)
        mask = mask.to(self.device)

        denominator = torch.exp(cosine_similarity) * mask
        denominator = torch.log(denominator.sum(axis=1))

        loss = - (nominator - denominator)  # - because of log

        return loss.mean()
