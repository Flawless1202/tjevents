import torch
import torch.nn as nn

from .base import BaseModel
from .unet import FireUNet


class FireNet(BaseModel):

    def __init__(self, args):
        super(FireNet, self).__init__(args)

        self.net = FireUNet(args)

    def forward(self, events, pre_states):
        img, states = self.net(events, pre_states)
        return img, states