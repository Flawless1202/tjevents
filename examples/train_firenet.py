import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision.transforms import Compose
from torch import optim
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from lpips_pytorch import lpips
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler

from tjevents.datasets import EventImageDataset
from tjevents.transforms import Event2VoxelGrid, VoxelGridPreprocess, ImageNormal, Pad
from tjevents.nn.models import FireNet
from tjevents.utils import parse_args
from tjevents.utils.types import as_easy_dict


class LightningFireNet(pl.LightningModule):

    def __init__(self, args):
        super(LightningFireNet, self).__init__()

        args = as_easy_dict(args)
        self.model_args = as_easy_dict(args.model)
        self.dataset_args = as_easy_dict(args.dataset)
        self.train_args = as_easy_dict(args.train)

        self.model = FireNet(self.model_args)

    def forward(self, events, pre_states):
        return self.model(events, pre_states)

    def training_step(self, train_batch, batch_idx):
        events, imgs = train_batch

        state = None
        loss = 0

        for seq in range(events.size()[1]):
            event = events[:, seq, ...]
            img = imgs[:, seq, ...]

            out, state = self(event, state)

            loss += torch.mean(lpips(out.repeat([1, 3, 1, 1]), img.unsqueeze(1).repeat([1, 3, 1, 1]),
                                     net_type="alex", version="0.1"))

        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, valid_batch, batch_idx):
        events, imgs = valid_batch

        state = None
        loss = 0

        for seq in range(events.size()[1]):
            event = events[:, seq, ...]
            img = imgs[:, seq, ...]

            out, state = self(event, state)

            loss += torch.mean(lpips(out.repeat([1, 3, 1, 1]), img.unsqueeze(1).repeat([1, 3, 1, 1]),
                                     net_type="alex", version="0.1"))

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

    def prepare_data(self):
        event_transform = Compose([
            Event2VoxelGrid(self.dataset_args.num_bins, self.dataset_args.width, self.dataset_args.height),
            VoxelGridPreprocess(self.dataset_args.if_normalize, self.dataset_args.flip),
            # Pad(self.dataset_args.width, self.dataset_args.height, self.dataset_args.size_divisor)
        ])

        img_transform = Compose([
            ImageNormal(),
            # Pad(self.dataset_args.width, self.dataset_args.height, self.dataset_args.size_divisor)
        ])

        self.train_dataset = EventImageDataset(self.dataset_args.data_root, "val",
                                               event_transform=event_transform, img_transform=img_transform)
        self.val_dataset = EventImageDataset(self.dataset_args.data_root, "val",
                                             event_transform=event_transform, img_transform=img_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_args.batch_size,
                          shuffle=self.train_args.shuffle, num_workers=self.train_args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_args.batch_size,
                          num_workers=self.train_args.num_workers)

    def configure_optimizers(self):
        optimizers = [optim.Adam(self.parameters(), lr=self.train_args.lr)]
        schedulers = [{
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizers[0], patience=3),
            "monitor": 'avg_val_loss',
            "interval": "epoch",
            "frequency": 1
        }]

        return optimizers, schedulers


def main():
    args = parse_args()
    model = LightningFireNet(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=args.checkpoint_path,
        save_top_k=3,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        prefix=''
    )

    logger = TensorBoardLogger(save_dir=args.log_path, name=args.name, version=args.version)
    profiler = AdvancedProfiler()

    trainer = pl.Trainer(gpus=model.train_args.num_gpus,
                         max_epochs=model.train_args.max_epochs,
                         logger=logger,
                         profiler=profiler,
                         weights_summary="full",
                         checkpoint_callback=checkpoint_callback,
                         resume_from_checkpoint=args.resume_from_checkpoint,
                         accumulate_grad_batches=model.train_args.batch_size_times)

    trainer.fit(model)


if __name__ == '__main__':
    main()
