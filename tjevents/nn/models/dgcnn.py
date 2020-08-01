import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import DynamicEdgeConv, global_max_pool


class DGCNN(nn.Module):

    def __init__(self, out_channels, k=8, aggr="max"):
        super(DGCNN, self).__init__()

        self.conv1 = DynamicEdgeConv(self._MLP([2 * 4, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(self._MLP([2 * 64, 128]), k, aggr)
        self.fc = self._MLP(([128 + 64, 256]))
        self.out = nn.Sequential(
            self._MLP([256, 128]), nn.Dropout(0.5),
            self._MLP([128, 64]), nn.Dropout(0.5),
            nn.Linear(64, out_channels)
        )

    def forward(self, data):
        pos, x, batch = data.pos, data.x, data.batch
        x1 = self.conv1(torch.cat([pos, x], dim=1))
        x2 = self.conv2(x1, batch)
        out = self.fc(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.out(out)
        return F.log_softmax(out, dim=1)

    @staticmethod
    def _MLP(channels, batch_norm=True):
        return nn.Sequential(*[
            nn.Sequential(nn.Linear(channels[i - 1],
                                    channels[i]),
                          nn.ReLU(),
                          nn.BatchNorm1d(channels[i]) if batch_norm else nn.Sequential())
            for i in range(1, len(channels))
        ])
