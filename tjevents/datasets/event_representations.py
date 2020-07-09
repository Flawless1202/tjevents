import torch

from tjevents.utils import CpuTimer, CudaTimer


class Event2Voxel(object):

    def __init__(self, num_bins, width, height):
        self.num_bins = num_bins
        self.width = width
        self.height = height

        self.Timer = CpuTimer

    def __call__(self, events):

        # with self.Timer("Events -> Device {}".format(self.device.type)):
        #     events_tensor = events_tensor.to(self.device)

        with self.Timer("Envents -> Voxel grid"):
            voxel = [self._event2voxel_single(torch.as_tensor(event)).unsqueeze(0) for event in events]

        return torch.cat(voxel, dim=0)

    def _event2voxel_single(self, events_tensor):

        voxel = torch.zeros(self.num_bins, self.height, self.width, dtype=torch.float32).flatten()

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
        voxel.index_add_(dim=0,
                         index=xs[valid_indices] + ys[valid_indices] * self.width +
                               tis.long()[valid_indices] * self.width * self.height,
                         source=vals_left[valid_indices])

        valid_indices = ((tis + 1) < self.num_bins) * (tis >= 0)
        voxel.index_add_(dim=0,
                         index=xs[valid_indices] + ys[valid_indices] * self.width +
                               (tis.long()[valid_indices] + 1) * self.width * self.height,
                         source=vals_right[valid_indices])

        voxel = voxel.view(self.num_bins, self.height, self.width)

        return voxel
