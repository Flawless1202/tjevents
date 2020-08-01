import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConvLayers(nn.Module):

    def __init__(self, conv, activation="relu", norm=None):
        super(BaseConvLayers, self).__init__()

        self.conv = conv
        self.activation = getattr(torch, activation, "relu") if activation is not None else nn.Sequential()

        assert norm in ["BN", "IN", None]
        if norm is None:
            self.norm = nn.Sequential()
        else:
            Norm = nn.BatchNorm2d if norm == "BN" else nn.InstanceNorm2d
            self.norm = Norm(conv.out_channels)

        self.out_channels = conv.out_channels

    @staticmethod
    def preprocess(x):
        return x

    def forward(self, x):
        x = self.preprocess(x)
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class ConvLayers(BaseConvLayers):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        bias = False if norm == "BN" else True
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        super(ConvLayers, self).__init__(conv, activation, norm)


class ConvTransposedLayers(BaseConvLayers):

    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, activation='relu', norm=None):
        bias = False if norm == "BN" else True
        conv_transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                             output_padding=1, bias=bias)
        super(ConvTransposedLayers, self).__init__(conv_transposed, activation, norm)


class UpsampleConvLayers(BaseConvLayers):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        bias = False if norm == "BN" else True
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        super(UpsampleConvLayers, self).__init__(conv, activation, norm)

    @staticmethod
    def preprocess(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm=None):
        super(ResidualBlock, self).__init__()

        bias = False if norm == "BN" else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)

        assert norm in ["BN", "IN", None]
        if norm is None:
            self.bn1, self.bn2 = nn.Sequential(), nn.Sequential()
        else:
            Norm = nn.BatchNorm2d if norm == "BN" else nn.InstanceNorm2d
            self.bn1, self.bn2 = Norm(out_channels), Norm(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)

        return out