#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-21 上午10:18
import torch
import pickle
import numpy as np
import os
import json
from tqdm import tqdm
from model.model import RNNModel
from torch.utils.data import DataLoader
from utils.datasets import DefinitionModelingDataset, DefinitionModelingCollate
# from train import get_testdata
# from source.pipeline import generate
from utils.datasets import Vocabulary
# from source.utils import prepare_ada_vectors_from_python, prepare_w2v_vectors
# from source.constants import BOS
import argparse

parser = argparse.ArgumentParser(description='Script to generate using model')
parser.add_argument(
    "--params", type=str, required=True,
    help="path to saved model params"
)
parser.add_argument(
    "--ckpt", type=str, required=True,
    help="path to saved model weights"
)
parser.add_argument(
    "--tau", type=float, required=True,
    help="temperature to use in sampling"
)
# parser.add_argument(
#     "--n", type=int, required=True,
#     help="number of samples to generate"
# )
parser.add_argument(
    "--length", type=int, required=True,
    help="maximum length of generated samples"
)
parser.add_argument(
    "--prefix", type=str, required=False,
    help="prefix to read until generation starts"
)
parser.add_argument(
    "--generate_list",type=str,required=True,
    help="path to word list to generate"
)
parser.add_argument(
    "--wordlist", type=str, required=False,
    help="path to word list with words and contexts"
)
parser.add_argument(
    "--w2v_binary_path", type=str, required=False,
    help="path to binary w2v file"
)
parser.add_argument(
    "--ada_binary_path", type=str, required=False,
    help="path to binary ada file"
)
parser.add_argument(
    "--prep_ada_path", type=str, required=False,
    help="path to prep_ada.jl script"
)
parser.add_argument(
    "--gen_dir", type=str, default="gen/",
    help="where to save generate file"
)
parser.add_argument(
    "--gen_name", type=str, default="gen.txt",
    help="generate file name"
)
args = parser.parse_args()

with open(args.params, "r") as infile:
    model_params = json.load(infile)

dataset = DefinitionModelingDataset(
    file=args.generate_list,
    vocab_path=model_params["voc"],
    input_vectors_path=model_params["input_test"],
    ch_vocab_path=model_params["ch_voc"],
    use_seed=model_params["use_seed"],
    hypm_path=model_params["hypm_test"],
    mode="gen"
)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=DefinitionModelingCollate,
    num_workers=2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNModel(model_params).to(device)
model.load_state_dict(torch.load(args.ckpt))
voc = Vocabulary()
voc.load(model_params["voc"])


def generate(model, dataloader, idx2word, strategy='greedy', max_len=50):
    model.training = False
    for inp in tqdm(dataloader, desc='Generate Definitions', leave=False):
        word_list = []
        data = {
            'word': torch.from_numpy(inp['word']).to(device),
            'seq': torch.t(torch.from_numpy(inp["seq"])).to(device),
        }
        if model.use_input:
            data["input_vectors"] = torch.from_numpy(inp['input']).to(device)
        if model.use_ch:
            data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
        if model.use_he:
            data["hypm"] = torch.from_numpy(inp['hypm']).long().to(device)
            data["hypm_weights"] = torch.from_numpy(inp['hypm_weights']).float().to(device)

        if not os.path.exists(args.gen_dir):
            os.makedirs(args.gen_dir)
        def_word = [idx2word[inp['word'][0]], "\t"]
        word_list.extend(def_word)
        hidden = None
        for i in range(max_len):
            output, hidden = model(data, hidden)
            word_weights = output.squeeze().div(args.tau).exp().cpu()
            if strategy == 'greedy':
                word_idx = torch.argmax(word_weights)
            elif strategy == 'multinomial':
                # 基于词的权重，对其再进行一次抽样，增添其多样性，如果不使用此法，会导致常用字的无限循环
                word_idx = torch.multinomial(word_weights, 1)[0]
            if word_idx == 3:
                break
            else:
                data['seq'].fill_(word_idx)
                word = idx2word[word_idx.item()]
                word_list.append(word)
        with open(args.gen_dir + args.gen_name, "a") as f:
            for item in word_list:
                f.write(item + " ")
            f.write("\n")
            f.close()
    print("Finished!")
    return 1


if __name__ == "__main__":
    print('=========model architecture==========')
    print(model)
    print('=============== end =================')
    generate(model, dataloader, voc.id2token, max_len=25)
