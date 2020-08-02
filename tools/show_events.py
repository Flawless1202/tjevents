import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from tjevents.datasets import Event2Voxel, VoxelPreprocess
from tjevents.utils import show_results, FixedDurationEventReader


root = "data/event_camera/"
subtype = "dynamic_6dof"
events_file = os.path.join(root, "origin", subtype, "events.txt")
image_list_file = os.path.join(root, "slomo", subtype, "images.txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_list = pd.read_csv(image_list_file, delim_whitespace=True, header=None,
                         names=["t", "image"],
                         dtype={"t": np.float64, "image": str},
                         engine="c",
                         nrows=None, memory_map=True).values

event_reader = FixedDurationEventReader(events_file)
transforms = Compose([Event2Voxel(5, 240, 180),
                      VoxelPreprocess(True, False)])

for event in event_reader:
    # mean_time_stamp = (event[-1, 0] + event[0, 0]) / 2
    img_idx = np.argmin(np.abs(image_list[:, 0] - event[-1, 0]), axis=0)
    voxel = transforms([event])
    img = cv2.imread(os.path.join(root, "slomo", subtype, image_list[img_idx, 1]))
    show_results(voxel, img)
