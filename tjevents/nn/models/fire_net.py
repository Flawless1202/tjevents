import torch
import torch.nn as nn

from .unet import FireUNet


class FireNet(nn.Module):

    def __init__(self, args):
        super(FireNet, self).__init__()

        self.net = FireUNet(args)

    def forward(self, events, pre_states):
        img, states = self.net(events, pre_states)
        return img, states
