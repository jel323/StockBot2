import torch

from torch.nn import MSELoss


class MSLE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = MSELoss()

    def forward(self, pred, actual):
        return self.mse(torch.log(pred), torch.log(actual))
