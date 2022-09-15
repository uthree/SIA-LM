import torch
import torch.nn as nn
import torch.optim as optim

from tokenizer import encode, decode
from model import SIALM

from tqdm import tqdm
from masked_word_dataset import MaskedWordDataset
from shuffle_word_dataset import ShuffleWordDataset
from predict_next_dataset import PredictNextDataset

import os

# training config
NUM_EPOCH = 10
INPUT_MAX_LEN = 512
OUTPUT_MAX_LEN = 128
BATCH_SIZE = 16

# initialize or load model
configs = {}
if os.path.exists('./model.pt'):
    print("Load Model")
    model = SIALM(**configs)
    model.load_state_dict(torch.load('./model.pt'))
else:
    print("Initialize model")
    model = SIALM(**configs)

# load Dataset
datasets = [
        #MaskedWordDataset(["/home/uthree/ao-childes-torch-dataset/aochildes.txt"]),
        #ShuffleWordDataset(["/home/uthree/ao-childes-torch-dataset/aochildes.txt"]),
        PredictNextDataset(["/mnt/f/logs/nucc_discord_special_tokens.txt"]),
        ]
dataset = torch.utils.data.ConcatDataset(datasets)

# training
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=256)
optimizer = optim.RAdam(model.parameters())
bar = tqdm(total=NUM_EPOCH*len(dataset))

for epoch in range(NUM_EPOCH):
    for batch, (src, tgt) in enumerate(dataloader):
        optimizer.zero_grad()
        #tqdm.write(f"{src}, {tgt}")
        src = torch.LongTensor([encode(s, max_len=INPUT_MAX_LEN) for s in src]).to(device)
        tgt = torch.LongTensor([encode(s, max_len=OUTPUT_MAX_LEN-1) for s in tgt])
        tgt_in = torch.cat([torch.full((tgt.shape[0], 1), 256, dtype=torch.int), tgt], dim=1).to(device)
        tgt_out = torch.cat([tgt, torch.full((tgt.shape[0], 1), 256, dtype=torch.int)], dim=1).to(device)
        out = model(src, tgt_in)
        loss = criterion(torch.flatten(out, start_dim=0, end_dim=1), torch.flatten(tgt_out, start_dim=0, end_dim=1))
        loss.backward()
        optimizer.step()
        if batch % 200 == 0:
            torch.save(model.state_dict(), './model.pt')
        desc = f"Loss: {loss.item():.6f}"
        bar.set_description(desc=desc)
        bar.update(n=src.shape[0])
