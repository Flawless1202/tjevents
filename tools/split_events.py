import os
import pickle as pkl
import gc

import cv2
import numpy as np
import pandas as pd
import torch

from tjevents.utils import FixedDurationEventReader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = "data/event_camera/"
origin_root = "data/event_camera/origin"

os.makedirs(os.path.join(root, "split_events"), exist_ok=True)
os.makedirs(os.path.join(root, "split_images"), exist_ok=True)

for subtype in os.listdir(origin_root):
    count = 0
    event_save = []
    img_save = []

    events_file = os.path.join(origin_root, subtype, "events.txt")
    image_list_file = os.path.join(root, "slomo",  subtype, "images.txt")
    image_list = pd.read_csv(image_list_file, delim_whitespace=True, header=None,
                             names=["t", "image"],
                             dtype={"t": np.float64, "image": str},
                             engine="c",
                             nrows=None, memory_map=True).values

    event_reader = FixedDurationEventReader(events_file)

    for event in event_reader:
        # if len(event) == 0 or event[-1, 0] - event[0, 0] < 0.04:
        #     break
        # mean_time_stamp = (event[-1, 0] + event[0, 0]) / 2
        img_idx = np.argmin(np.abs(image_list[:, 0] - event[-1, 0]), axis=0)
        img = cv2.imread(os.path.join(root, "slomo", subtype, image_list[img_idx, 1]), cv2.IMREAD_GRAYSCALE)

        event_save.append(event)
        img_save.append(img)

        count += 1

        if len(event_save) == 30:
            with open(os.path.join(root, "events", "{}_{}.pkl".format(subtype, count)), "wb") as f:
                pkl.dump(event_save, f)
            with open(os.path.join(root, "images", "{}_{}.pkl".format(subtype, count)), "wb") as f:
                pkl.dump(img_save, f)

            del event_save, img_save
            gc.collect()

            event_save = []
            img_save = []

    del event_reader
    gc.collect()
