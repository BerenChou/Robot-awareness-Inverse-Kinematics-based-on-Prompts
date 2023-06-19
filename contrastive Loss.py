import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    # TODO
    def forward(self):
        pass
