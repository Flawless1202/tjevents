import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from tjevents.datasets import Event2Voxel, VoxelPreprocess
from tjevents.utils import show_results, FixedDurationEventReader


root = "data/event_camera/dynamic_6dof"
events_file = os.path.join(root, "events.txt")
image_list_file = os.path.join(root, "images.txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_list = pd.read_csv(image_list_file, delim_whitespace=True, header=None,
                         names=["t", "image"],
                         dtype={"t": np.float64, "image": str},
                         engine="c",
                         nrows=None, memory_map=True).values

event_reader = FixedDurationEventReader(events_file)
transforms = Compose([Event2Voxel(5, 240, 180, device),
                      VoxelPreprocess(True, False, device)])

for event in event_reader:
    mean_time_stamp = (event[-1, 0] + event[0, 0]) / 2
    img_idx = np.argmin(np.abs(image_list[:, 0] - mean_time_stamp), axis=0)
    voxel = transforms(event).unsqueeze(0)
    img = cv2.imread(os.path.join(root, image_list[img_idx, 1]))
    show_results(voxel, img)
