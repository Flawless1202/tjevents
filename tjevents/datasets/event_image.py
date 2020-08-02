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

    def __init__(self, root, event_transform=None, img_transform=None):
        super(EventImageDataset, self).__init__()

        self.all_event_files = glob.glob(os.path.join(root, "event", "*"))

        # self.events_trans = Compose([
        #     Event2Voxel(self.args.num_bins, self.args.width, self.args.height),
        #     VoxelPreprocess(self.args.if_normalize, self.args.flip),
        #     # Pad(12)
        # ])
        self.event_transform = event_transform

        # self.img_trans = Compose([
        #     ImageNormal(),
        #     # Pad(12)
        # ])
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

        return self.event_transform(event), self.img_transform(imgs)

    @staticmethod
    def _get_img_by_event(event_file):
        return event_file.replace("events", "image")
