#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-14 下午8:54
import numpy as np
import argparse
import time
import json
import pickle
from collections import defaultdict
from utils.datasets import Vocabulary
from utils.util import get_time_dif, read_data

parser = argparse.ArgumentParser(description="Prepare hypernyms data for model.")
parser.add_argument(
    '--defs', type=str, required=True, nargs="+",
    help='location of txt file with definitions.'
)
parser.add_argument(
    '--save', type=str, required=True, nargs="+",
    help='where to save files'
)
parser.add_argument(
    '--hypm', type=str, required=True,
    help="location of bag of hypernyms."
)
parser.add_argument(
    '--vocab', type=str, required=True,
    help="preprocessed vocabulary json file."
)
parser.add_argument(
    '--top_k', type=int, default=5,
    help="numbers of hypernyms to use."
)
parser.add_argument(
    '--save_hypm', type=str, default="../data/processed/word2hypm.json",
    help="where to save word hypernyms"
)
parser.add_argument(
    '--save_weights', type=str, default="../data/processed/hypm_weights.json",
    help="where to save word hypernyms weights"
)
args = parser.parse_args()
if len(args.defs) != len(args.save):
    parser.error("Number of defs files must match number of save locations.")


def read_hypernyms(file_path):
    """Read hypernyms"""
    hnym_data = defaultdict(list)
    with open(file_path, 'r') as fr:
        for line in fr:
            line = line.strip().split('\t')
            word = line[0]
            line = line[1:]
            assert len(line) % 2 == 0
            for i in range(int(len(line) / 2)):
                hnym = line[2 * i]
                weight = line[2 * i + 1]
                hnym_data[word].append((hnym, weight))
    return hnym_data


def get_hnym(hnym_data, vocab):
    """Get hypernyms and weights"""
    word2hnym = defaultdict(list)
    hnym_weights = defaultdict(list)
    for key, value in hnym_data.items():
        weight_sum = sum([float(w) for h, w in value])
        for hnym, weight in value:
            word2hnym[key].append(vocab.encode(hnym))
            hnym_weights[key].append(float(weight) / weight_sum)
    return word2hnym, hnym_weights


if __name__ == "__main__":
    start_time = time.time()
    print('Start prepare word hypernyms and weights at {}'.format(time.asctime(time.localtime(start_time))))
    hypernym_data = read_hypernyms(args.hypm)
    vocab = Vocabulary()
    vocab.load(args.vocab)
    word2hym, hym_weights = get_hnym(hypernym_data, vocab)
    for i in range(len(args.defs)):
        data = read_data(args.defs[i])
        hnym = np.zeros((len(data), args.top_k))
        hnym_weights = np.zeros_like(hnym)
        assert len(data) == len(hnym)
        for l, (word, _) in enumerate(data):
            for j, h in enumerate(word2hym[word][:args.top_k]):
                hnym[l][j] = h
            for k, weight in enumerate(hym_weights[word][:args.top_k]):
                hnym_weights[l][k] = weight
        with open(args.save[i], 'wb') as outfile:
            pickle.dump([hnym, hnym_weights], outfile)
            outfile.close()
    with open(args.save_hypm, 'w') as f:
        json.dump(word2hym, f, indent=4)
        f.close()
    with open(args.save_weights, 'w') as f:
        json.dump(hym_weights, f, indent=4)
        f.close()
    time_dif = get_time_dif(start_time)
    print("Finished!Prepare word hypernyms time usage:", time_dif)
