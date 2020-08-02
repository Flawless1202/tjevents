import re
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster


class Pad(object):

    def __init__(self, width, height, size_divisor=32):
        padded_height = self._get_padded_size(height, size_divisor)
        padded_width = self._get_padded_size(width, size_divisor)
        
        padding_top = ceil((padded_height - height))
        padding_bottom = padded_height - padding_top
        padding_left = ceil((padded_width - width))
        padding_right = padded_width - padding_left

        self.pad = nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)

    def __call__(self, x):
        return self.pad(x)

    @staticmethod
    def _get_padded_size(origin_size, size_divisor):
        return int(size_divisor * ceil(origin_size / size_divisor))


class ImageNormal(object):

    def __call__(self, img):
        return img.float().div(255.0)


class Event2VoxelGrid(object):
    """ The event representation from the `"High Speed and High Dynamic Range Videowith an
    Event Camera" <https://arxiv.org/pdf/1906.07165>`_ paper.

    """

    def __init__(self, num_bins, width, height):
        self.num_bins = num_bins
        self.width = width
        self.height = height

    def __call__(self, events):
        voxel = [self._event2voxel_grid_single(torch.as_tensor(event)).unsqueeze(0) for event in events]

        return torch.cat(voxel, dim=0)

    def _event2voxel_grid_single(self, events_tensor):
        voxel_grid = torch.zeros(self.num_bins, self.height, self.width, dtype=torch.float32).flatten()

        delta_t = events_tensor[-1, 0] - events_tensor[0, 0]
        delta_t = 1.0 if abs(delta_t) < 1e-9 else delta_t

        events_tensor[:, 0] = (self.num_bins - 1) * (events_tensor[:, 0] - events_tensor[0, 0]) / delta_t
        ts, xs, ys, pols = [events_tensor[:, idx] for idx in range(4)]
        xs, ys = xs.long(), ys.long()
        pols = (pols * 2 - 1).float()

        tis = torch.floor(ts)
        dts = (ts - tis).float()
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts

        valid_indices = (tis < self.num_bins) * (tis >= 0)
        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * self.width +
                                    tis.long()[valid_indices] * self.width * self.height,
                              source=vals_left[valid_indices])

        valid_indices = ((tis + 1) < self.num_bins) * (tis >= 0)
        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * self.width +
                                    (tis.long()[valid_indices] + 1) * self.width * self.height,
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(self.num_bins, self.height, self.width)

        return voxel_grid


class VoxelGridPreprocess(object):

    def __init__(self, if_normalize, flip):
        self.if_normalize = if_normalize
        self.flip = flip

    def __call__(self, voxel_grid):

        if self.flip:
            voxel_grid = torch.flip(voxel_grid, dims=[2, 3])

        if self.if_normalize:
            nonzero_voxel_grid = (voxel_grid != 0.)
            num_nonzero_voxel_grid = nonzero_voxel_grid.sum()

            if num_nonzero_voxel_grid > 0:
                mean = voxel_grid.sum() / num_nonzero_voxel_grid
                stddev = torch.sqrt((voxel_grid ** 2).sum() / num_nonzero_voxel_grid - mean ** 2)
                voxel_grid = nonzero_voxel_grid.float() * (voxel_grid - mean) / stddev

        return voxel_grid


class EventGridSampling(object):
    """Clusters points into voxels with size :attr:`size`. Modified from the `torch_geometric.transforms.GridSampling`

    Args:
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
    """
    def __init__(self, size, start=None, end=None, keep_pol=True):
        self.size = size
        self.start = start
        self.end = end
        self.keep_pol = keep_pol

    def __call__(self, data):
        num_nodes = data.num_nodes

        if 'batch' not in data:
            batch = data.pos.new_zeros(num_nodes, dtype=torch.long)
        else:
            batch = data.batch

        cluster = voxel_grid(data.pos, batch, self.size, self.start, self.end)
        cluster, perm = consecutive_cluster(cluster)

        for key, item in data:
            if bool(re.search('edge', key)):
                raise ValueError(
                    'GridSampling does not support coarsening of edges')

            if torch.is_tensor(item) and item.size(0) == num_nodes:
                if key == 'y':
                    item = F.one_hot(item)
                    item = scatter_add(item, cluster, dim=0)
                    data[key] = item.argmax(dim=-1)
                elif key == 'batch':
                    data[key] = item[perm]
                elif key == 'x' and self.keep_pol:
                    data[key] = item[perm]
                else:
                    data[key] = scatter_mean(item, cluster, dim=0)

        return data

    def __repr__(self):
        return '{}(size={})'.format(self.__class__.__name__, self.size)
