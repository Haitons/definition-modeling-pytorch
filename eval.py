#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-21 下午2:40
import os
import pickle
import torch
import numpy as np
import json
from torch import nn
from tqdm import tqdm
from model.model import RNNModel
from torch.utils.data import DataLoader
from utils.datasets import DefinitionModelingDataset, DefinitionModelingCollate
from utils.datasets import Vocabulary
# from source.constants import BOS
# from source.pipeline import test
# from source.pipeline import generate
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import torch

parser = argparse.ArgumentParser(description='Script to evaluate model')
parser.add_argument(
    "--params", type=str, required=True,
    help="path to saved model params"
)
parser.add_argument(
    "--ckpt", type=str, required=True,
    help="path to saved model weights"
)
parser.add_argument(
    "--datasplit", type=str, required=True,
    help="train, val or test set to evaluate on"
)
parser.add_argument(
    "--type", type=str, required=True,
    help="compute score or bleu"
)
parser.add_argument(
    "--wordlist", type=str, required=False,
    help="word list to evaluate on (by default all data will be used)"
)
# params for BLEU
parser.add_argument(
    "--tau", type=float, required=False,
    help="temperature to use in sampling"
)
parser.add_argument(
    "--n", type=int, required=False,
    help="number of samples to generate"
)
parser.add_argument(
    "--length", type=int, required=False,
    help="maximum length of generated samples"
)
args = parser.parse_args()
assert args.datasplit in ["train", "val", "test"], ("--datasplit must be "
                                                    "train, val or test")
assert args.type in ["score", "bleu"], ("--type must be score or bleu")

with open(args.params, "r") as infile:
    model_params = json.load(infile)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNModel(model_params).to(device)
model.load_state_dict(torch.load(args.ckpt))

if args.datasplit == "train":
    dataset = DefinitionModelingDataset(
        file=model_params["train_defs"],
        vocab_path=model_params["voc"],
        input_vectors_path=model_params["input_train"],
        ch_vocab_path=model_params["ch_voc"],
        use_seed=model_params["use_seed"],
        hypm_path=model_params["hypm_train"]
    )
elif args.datasplit == "val":
    dataset = DefinitionModelingDataset(
        file=model_params["eval_defs"],
        vocab_path=model_params["voc"],
        input_vectors_path=model_params["input_eval"],
        ch_vocab_path=model_params["ch_voc"],
        use_seed=model_params["use_seed"],
        hypm_path=model_params["hypm_eval"]
    )
elif args.datasplit == "test":
    dataset = DefinitionModelingDataset(
        file=model_params["test_defs"],
        vocab_path=model_params["voc"],
        input_vectors_path=model_params["input_test"],
        ch_vocab_path=model_params["ch_voc"],
        use_seed=model_params["use_seed"],
        hypm_path=model_params["hypm_test"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1 if args.type == "bleu" else model_params["batch_size"],
        collate_fn=DefinitionModelingCollate,
        shuffle=True,
        num_workers=2
    )


def test(model, dataloader, device):
    model.training = False
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        total_loss = []
        for inp in tqdm(dataloader):
            data = {
                'word': torch.from_numpy(inp['word']).to(device),
                'seq': torch.t(torch.from_numpy(inp['seq'])).to(device),
            }
            if model.use_input:
                data["input_vectors"] = torch.from_numpy(inp['input']).to(device)
            if model.use_ch:
                data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
            if model.use_he:
                data["hypm"] = torch.from_numpy(inp['hypm']).long().to(device)
                data["hypm_weights"] = torch.from_numpy(inp['hypm_weights']).float().to(device)
            targets = torch.t(torch.from_numpy(inp['target'])).to(device)
            output, hidden = model(data, None)
            loss = loss_fn(output, targets.contiguous().view(-1))
            total_loss.append(loss.item())
    return np.mean(total_loss), np.exp(np.mean(total_loss))


if __name__ == '__main__':
    print('=========model architecture==========')
    print(model)
    print('=============== end =================')
    if args.type == "score":
        loss, ppl = test(model, dataloader, device)
    print("The test set Loss:{0:>6.6},Ppl:{1:>6.6}".format(loss, ppl))
