import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch import optim
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler

from tjevents.datasets import E2GDataset
from tjevents.nn.models import DGCNN
from tjevents.utils import parse_args
from tjevents.utils.types import as_easy_dict


class LightningDGCNN(pl.LightningModule):
    
    def __init__(self, args):
        super(LightningDGCNN, self).__init__()

        args = as_easy_dict(args)
        self.model_args = as_easy_dict(args.model)
        self.dataset_args = as_easy_dict(args.dataset)
        self.train_args = as_easy_dict(args.train)

        self.model = DGCNN(self.model_args)

    def forward(self, data):
        return self.model(data)

    def training_step(self, train_batch, batch_idx):
        data = train_batch
        logits = self(data)
        loss = F.nll_loss(logits, data.y)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, valid_batch, batch_idx):
        data = valid_batch
        logits = self(data)
        loss = F.nll_loss(logits, data.y)
        corrects = logits.detach().max(1)[1].eq(data.y).sum()
        return {"val_loss": loss, "val_corrects": corrects}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_corrects"] for x in outputs]).sum().float() / \
                  len(outputs) * self.dataset_args.batch_size
        logs = {"val_loss": avg_loss, "val_accuracy": avg_acc}
        print(avg_acc)
        return {"avg_val_loss": avg_loss, "log": logs}

    def prepare_data(self):
        e2g_train = E2GDataset(self.dataset_args.data_root, transform=T.NormalizeScale())
        val_len = int(len(e2g_train) * self.train_args.train_val_split)
        self.train_dataset, self.val_dataset = random_split(e2g_train, [len(e2g_train) - val_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.dataset_args.batch_size,
                          shuffle=self.dataset_args.shuffle, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.dataset_args.batch_size, num_workers=12)

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
    model = LightningDGCNN(args)

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
                         checkpoint_callback=checkpoint_callback,
                         resume_from_checkpoint=args.resume_from_checkpoint,
                         limit_train_batches=0.01,
                         limit_val_batches=0.01)

    trainer.fit(model)


if __name__ == '__main__':
    main()
