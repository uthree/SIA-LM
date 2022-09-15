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
OUTPUT_MAX_LEN = 256
BATCH_SIZE = 16

# initialize or load model
configs = {}
if os.path.exists('./model.pt'):
    print("Load Model")
    model = SIALM(**configs)
    model.load_state_dict(torch.load('./model.pt', map_location=torch.device('cpu')))
else:
    print("Initialize model")
    model = SIALM(**configs)


MAX_LOG_LENGTH=5
LOGO = """
   _____ _____
  / ____|_   _|   /\\
 | (___   | |    /  \\
  \___ \  | |   / /\ \\
  ____) |_| |_ / ____ \\
 |_____/|_____/_/    \_\\

SIA: Systematic Inteligence Assistant
Version 0.0.1, Lang=JaJp
"""
print(LOGO)
logs = []
while True:
    user_utterance = input("USER >") + "<sep>"
    logs.append(user_utterance)
    bot_utterance = model.predict_sentence("<sep>".join(logs[-MAX_LOG_LENGTH:]))
    logs.append(bot_utterance)
    s = bot_utterance
    s = s.replace("<sep>", "")
    s = s.replace("<laugh>", "笑")
    print("BOT >" + s)
