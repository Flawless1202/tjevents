import torch

from tjevents.utils import CpuTimer, CudaTimer


class Pad(object):

    def __init__(self, padding, fill=0):
        self.padding = padding

    def __call__(self, x):
        pad_size = list(x.size())
        pad_size[-2] = self.padding
        pad = torch.zeros(pad_size)

        return torch.cat([x, pad], dim=-2)




class ImageNormal(object):

    def __call__(self, img):
        return img.float().div(255.0)


class VoxelPreprocess(object):

    def __init__(self, if_normalize, flip):
        self.if_normalize = if_normalize
        self.flip = flip

        self.Timer = CpuTimer

    def __call__(self, voxel):

        if self.flip:
            voxel = torch.flip(voxel, dims=[2, 3])

        if self.if_normalize:
            with self.Timer("Normalize the voxel grid ..."):
                nonzero_voxel = (voxel != 0.)
                num_nonzero_voxel = nonzero_voxel.sum()

                if num_nonzero_voxel > 0:
                    mean = voxel.sum() / num_nonzero_voxel
                    stddev = torch.sqrt((voxel ** 2).sum() / num_nonzero_voxel - mean ** 2)
                    voxel = nonzero_voxel.float() * (voxel - mean) / stddev

        return voxel
