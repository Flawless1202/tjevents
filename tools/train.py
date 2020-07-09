import tqdm
from easydict import EasyDict

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from lpips_pytorch import LPIPS

from tjevents.config.config import parse_args
from tjevents.datasets import EventDataset
from tjevents.model import RecurrentE2VID

args = parse_args()
train_args = EasyDict(args.train)
device = torch.device("cuda" if torch.cuda.is_available() and train_args.cuda else "cpu")

train_dataset = EventDataset(args.data, "train")
train_loader = DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True,
                          num_workers=train_args.batch_size)

model = RecurrentE2VID(args).to(device)
lpips = LPIPS(net_type="vgg", version="0.1").to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(train_args.epochs):
    model.train()
    for events, imgs in train_loader:
        optimizer.zero_grad()

        state = None
        loss = 0
        for seq in range(events.size()[1]):
            event = events[:, seq, ...].to(device)
            img = imgs[:, seq, ...].to(device)

            out, state = model(event, state)

            loss += lpips(out, img)

        loss.backward()
        optimizer.step()

        print(loss)

