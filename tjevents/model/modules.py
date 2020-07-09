import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_rnn import ConvGRU, ConvLSTM


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


class BaseRecurrentLayers(nn.Module):

    def __init__(self, convs, rnn_type="ConvLSTM", downsample=False):
        super(BaseRecurrentLayers, self).__init__()

        assert rnn_type in ["ConvLSTM", "ConvGRU"]
        self.rnn_type = rnn_type
        rnn = ConvLSTM if rnn_type == "ConvLSTM" else ConvGRU
        self.rnn = rnn(input_size=convs.out_channels, hidden_size=convs.out_channels, kernel_size=3)

        self.convs = convs
        self.downsample = downsample

    def forward(self, x, state):
        x = self.convs(x)
        state = self.rnn(x, state)
        out = state[0] if self.rnn_type == "ConvLSTM" else state
        if self.downsample:
            out = self.convs.activation(F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False))
        return out, state


class RecurrentConvLayers(BaseRecurrentLayers):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 rnn_type="ConvLSTM", activation="relu", norm=None):
        convs = ConvLayers(in_channels, out_channels, kernel_size, stride, padding, activation, norm)
        super(RecurrentConvLayers, self).__init__(convs, rnn_type)


class DownsampleRecurrentConvLayers(BaseRecurrentLayers):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 rnn_type="ConvLSTM", activation="relu", norm=None):
        convs = ConvLayers(in_channels, out_channels, kernel_size, stride, padding, activation, norm)
        super(DownsampleRecurrentConvLayers, self).__init__(convs, rnn_type, downsample=True)


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


class RecurrentResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 recurrent_block_type='convlstm', norm=None):
        super(RecurrentResidualBlock, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type

        if self.rnn_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ResidualBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  norm=norm)
        self.recurrent_block = RecurrentBlock(input_size=out_channels,
                                              hidden_size=out_channels,
                                              kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state
