import torch
import torch.nn as nn

from .base import BaseModel
from .unet import RecurrentUNet


class RecurrentE2VID(BaseModel):

    def __init__(self, args):
        super(RecurrentE2VID, self).__init__(args)

        self.unet = RecurrentUNet(args)

    def forward(self, events, pre_states):
        img, states = self.unet(events, pre_states)
        return img, states