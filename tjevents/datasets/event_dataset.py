import os
import glob
from easydict import EasyDict
import pickle as pkl

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from .event_representations import Event2Voxel
from .transforms import VoxelPreprocess, ImageNormal, Pad


class EventDataset(Dataset):

    def __init__(self, args, phase):
        super(EventDataset, self).__init__()

        self.args = EasyDict(args)
        self.all_events_files = glob.glob(os.path.join(self.args.events_dir, phase, "*"))

        self.events_trans = Compose([
            Event2Voxel(self.args.num_bins, self.args.width, self.args.height),
            VoxelPreprocess(self.args.if_normalize, self.args.flip),
            Pad(12)
        ])

        self.img_trans = Compose([
            ImageNormal(),
            Pad(12)
        ])

    def __len__(self):
        return len(self.all_events_files)

    def __getitem__(self, idx):
        events_file = self.all_events_files[idx]
        with open(events_file, "rb") as f:
            events = pkl.load(f)

        img_file = self._get_img_by_event(self.all_events_files[idx])
        with open(img_file, "rb") as f:
            imgs = torch.cat([torch.as_tensor(img).unsqueeze(0) for img in pkl.load(f)], dim=0)

        return self.events_trans(events), self.img_trans(imgs)

    @staticmethod
    def _get_img_by_event(event_file):
        return event_file.replace("split_events", "split_images")