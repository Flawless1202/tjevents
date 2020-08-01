import torch
import torch.nn as nn
from torch.nn import init


class ConvLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        padding = kernel_size // 2

        self.gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=padding)

    def forward(self, x, state):

        batch_size = x.size()[0]
        spatial_size = x.size()[2:]

        if state is None:
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            state = (torch.zeros(state_size).to(x.device), torch.zeros(state_size).to(x.device))

        pre_hidden, pre_cell = state

        input = torch.cat((x, pre_hidden), 1)
        gates = self.gates(input)
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * pre_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRU, self).__init__()

        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate, self.update_gate, self.out_gate = \
            [nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding) for _ in range(3)]

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                init.constant_(m.bias, 0.)

    def forward(self, x, state):
        batch_size = x.size()[0]
        spatial_size = x.size()[2:]

        if state is None:
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            state = torch.zeros(state_size).to(x.device)

        input = torch.cat([x, state], dim=1)
        update = torch.sigmoid(self.update_gate(input))
        reset = torch.sigmoid(self.reset_gate(input))
        out = torch.tanh(self.out_gate(torch.cat([x, state * reset], dim=1)))
        state = state * (1 - update) + out * update

        return state
