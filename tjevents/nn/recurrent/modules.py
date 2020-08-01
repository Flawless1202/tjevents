import torch
import torch.nn as nn
import torch.nn.functional as F

from tjevents.nn.conv import ConvLayers, ResidualBlock

from .conv_rnn import ConvGRU, ConvLSTM


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


class RecurrentResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, rnn_type='ConvLSTM', norm=None):
        super(RecurrentResidualBlock, self).__init__()

        assert(rnn_type in ['ConvLSTM', 'ConvGRU'])
        self.rnn_type = rnn_type

        if self.rnn_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU

        self.conv = ResidualBlock(in_channels=in_channels, out_channels=out_channels,
                                  norm=norm)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.rnn_type == 'ConvLSTM' else state
        return x, state
