from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ConvLayers, UpsampleConvLayers, ConvTransposedLayers, RecurrentConvLayers, ResidualBlock, \
    RecurrentResidualBlock


class BaseUNet(nn.Module):

    def __init__(self, args):
        super(BaseUNet, self).__init__()

        self.args = EasyDict(args)

        skip_sum = lambda x1, x2: x1 + x2
        skip_concat = lambda x1, x2: torch.cat([x1, x2], dim=1)
        self.apply_skip = skip_sum if self.args.skip_type == "sum" else skip_concat

        self.up_layers = UpsampleConvLayers if self.args.use_upsample_conv else ConvTransposedLayers
        self.max_channels = self.args.base_channels * pow(2, self.args.num_encoders)

        self.encoder_input_sizes = [self.args.base_channels * pow(2, i) for i in range(self.args.num_encoders)]
        self.encoder_output_sizes = [self.args.base_channels * pow(2, i + 1) for i in range(self.args.num_encoders)]

        self.activation = getattr(torch, self.args.activation, "sigmoid")

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(self.max_channels, self.max_channels, norm=self.args.norm)]
        )

        decoder_input_sizes = list(reversed([self.args.base_channels * pow(2, i + 1)
                                             for i in range(self.args.num_encoders)]))
        self.decoders = nn.Sequential(
            *[self.up_layers(in_size if self.args.skip_type == "sum" else 2 * in_size,
                             in_size // 2, kernel_size=5, padding=2, norm=self.args.norm)
              for in_size in decoder_input_sizes]
        )

        self.pred = ConvLayers(self.args.base_channels if self.args.skip_type == "sum" else 2 * self.args.base_channels,
                               self.args.out_channels, kernel_size=1, activation=None, norm=self.args.norm)


class RecurrentUNet(BaseUNet):

    def __init__(self, args):
        super(RecurrentUNet, self).__init__(args)

        self.args = EasyDict(args)

        self.head = ConvLayers(self.args.in_channels, self.args.base_channels, kernel_size=5, stride=1, padding=2)
        self.encoders = nn.Sequential(
            *[RecurrentConvLayers(in_size, out_size, kernel_size=5, stride=2, padding=2,
                                  rnn_type=self.args.rnn_type, norm=self.args.norm)
              for in_size, out_size in zip(self.encoder_input_sizes, self.encoder_output_sizes)]
        )

    def forward(self, x, pre_states):
        x = self.head(x)
        head = x

        if pre_states is None:
            pre_states = [None] * self.args.num_encoders

        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, pre_states[i])
            blocks.append(x)
            states.append(state)

        for res_block in self.res_blocks:
            x = res_block(x)

        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip(x, blocks[self.args.num_encoders - i - 1]))

        img = self.activation(self.pred(self.apply_skip(x, head)))

        return img, states





