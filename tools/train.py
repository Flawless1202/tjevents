from tqdm import tqdm
from easydict import EasyDict

import torch
import torch.nn as nn
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

model = RecurrentE2VID(args)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).to(device)
lpips = LPIPS(net_type="vgg", version="0.1")
lpips = nn.DataParallel(lpips, device_ids=[0, 1, 2, 3]).to(device)
optimizer = Adam(model.parameters(), lr=train_args.lr)

for epoch in range(train_args.epochs):
    train_loss = 0
    train_count = 0
    model.train()
    for events, imgs in tqdm(train_loader, desc="Epoch {}/{} Train".format(epoch + 1, train_args.epochs)):
        optimizer.zero_grad()

        state = None
        loss = 0
        for seq in range(events.size()[1]):
            event = events[:, seq, ...].to(device)
            img = imgs[:, seq, ...].to(device)

            out, state = model(event, state)

            loss += torch.mean(lpips(out.repeat([1, 3, 1, 1]), img.unsqueeze(1).repeat([1, 3, 1, 1])))

        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()
        train_count += 1
        
        if train_count % 20 == 0:
            print("Epoch {} / {} - Train Loss: {}".format(epoch + 1, train_args.epochs, train_loss / train_count))

