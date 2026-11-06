import torch
from torch import nn


class ContactTimeFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("centers", torch.tensor([.6, .4, .25, .15, .08, .04, 0., -.04]))
        self.register_buffer("widths", torch.tensor([.1, .08, .06, .04, .025, .02, .015, .02]))

    def forward(self, state):
        goals = state[..., 189:269].reshape(*state.shape[:-1], 8, 10)
        distance = (goals[..., 7, None]-self.centers)/self.widths
        return (torch.exp(-.5*distance.square())*goals[..., 9, None]).flatten(start_dim=-2)


