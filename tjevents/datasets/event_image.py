import os
import glob
import pickle as pkl

import torch
from torch.utils.data import Dataset


class EventImageDataset(Dataset):
    """ The Event reconstruct to image dataset. The dataset dir should be organized as follow:
        dataset_root/
        ├── train/
        │   ├── event/
        │   │   ├── event1.txt
        │   │   ├── ...
        │   │   └── eventN.txt
        │   └── image
        │       ├── image1.png
        │       ├── ...
        │       └── imageN.png
        ├── val/
        │   ├── event/
        │   └── image/
        └── test/
            ├── event/
            └── image/
    """

    def __init__(self, root, phase, event_transform=None, img_transform=None):
        super(EventImageDataset, self).__init__()

        self.all_event_files = glob.glob(os.path.join(root, "events", phase,  "*"))

        self.event_transform = event_transform
        self.img_transform = img_transform

    def __len__(self):
        return len(self.all_event_files)

    def __getitem__(self, idx):
        event_file = self.all_event_files[idx]
        with open(event_file, "rb") as f:
            event = pkl.load(f)

        img_file = self._get_img_by_event(self.all_event_files[idx])
        with open(img_file, "rb") as f:
            imgs = torch.cat([torch.as_tensor(img).unsqueeze(0) for img in pkl.load(f)], dim=0)

        if self.event_transform is not None:
            event = self.event_transform(event)

        if self.img_transform is not None:
            imgs = self.img_transform(imgs)

        return event, imgs

    @staticmethod
    def _get_img_by_event(event_file):
        return event_file.replace("events", "images")
