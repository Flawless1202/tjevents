import os

import numpy as np
from scipy import io

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import radius_graph

from tjevents.transforms import GridSampling


class E2GDataset(Dataset):
    """ The Event to Graph dataset used in `"Graph-Based Object Classification for Neuromorphic Vision Sensing"
    <https://openaccess.thecvf.com/content_ICCV_2019/html/Bi_Graph-Based_Object_Classification_for_Neuromorphic_
    Vision_Sensing_ICCV_2019_paper.html>`_paper. GridSampling is used to downsampling the events.
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(E2GDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        file_names = os.listdir(self.raw_dir)
        return file_names

    @property
    def processed_file_names(self):
        file_names = [f.replace(".mat", ".pt") for f in os.listdir(self.raw_dir)]
        return file_names

    def download(self):
        return

    def process(self):
        for raw_path in self.raw_paths:
            events = io.loadmat(raw_path)
            events = np.concatenate((events["ts"] / 1000.0, events["x"],
                                     events["y"], events["pol"]), axis=1).astype(np.float32)
            events = torch.from_numpy(events)
            label = ord(raw_path.split("/")[-1][0]) - 96

            graph = Data(x=events[:, 3].view(-1, 1), y=label, pos=events[:, 0:3])
            grid_sampling = GridSampling(size=[16, 16, 16])
            grid_sampling(graph)
            graph.edge_index = radius_graph(graph.pos, 9)

            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            torch.save(graph, os.path.join(self.processed_dir, raw_path.split("/")[-1].replace("mat", "pt")))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data
