import os
from argparse import Namespace
import torch
import torch.nn as nn
import cv2

from .flownet2.resample2d_package.resample2d import Resample2d
from .flownet2 import FlowNet2


class FlowNet(nn.Module):

    def __init__(self):
        super(FlowNet, self).__init__()

        args = Namespace(fp16=False, rgb_max=1.)
        self.model = FlowNet2(args).cuda()

        checkpoint_path = os.path.join("/home/chenkai/.cache/torch/checkpoints", "FlowNet2_checkpoint.pth.tar")
        self.model.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    def forward(self, img1, img2):
        assert img1.device.type == "cuda" and img2.device.type == "cuda"
        return self.model(img1, img2)


class FlowWarper(nn.Module):

    def __init__(self):
        super(FlowWarper, self).__init__()

        self.warper = Resample2d().cuda()

    def forward(self, img, flow):
        return self.warper(img, flow)
