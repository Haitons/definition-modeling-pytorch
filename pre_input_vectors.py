#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-20 上午9:37
import time
import argparse
import pickle
import numpy as np
from utils.datasets import Vocabulary
from utils.util import get_time_dif, read_data

parser = argparse.ArgumentParser(description="Prepare word vectors for Input conditioning.")
parser.add_argument(
    '--defs', type=str, required=True, nargs="+",
    help='location of txt file with definitions.'
)

parser.add_argument(
    '--save', type=str, required=True, nargs="+",
    help='where to save files'
)
parser.add_argument(
    '--vocab', type=str, required=True,
    help="preprocessed vocabulary json file."
)
parser.add_argument(
    "--embedding", type=str, required=True,
    help="location of prepared word embedding file"
)
args = parser.parse_args()
if len(args.defs) != 3:
    parser.error("--defs must have both train,valid and test definitions file.")
if len(args.defs) != len(args.save):
    parser.error("Number of defs files must match number of save locations.")

if __name__ == '__main__':
    start_time = time.time()
    print('Start prepare input vectors at {}'.format(time.asctime(time.localtime(start_time))))
    for i in range(len(args.defs)):
        vectors = []
        data = read_data(args.defs[i])
        vocab = Vocabulary()
        vocab.load(args.vocab)
        with open(args.embedding, 'rb') as infile:
            word_embedding = pickle.load(infile)
        for element in data:
            vectors.append(word_embedding[vocab.encode(element[0])])
        with open(args.save[i], 'wb') as outfile:
            pickle.dump(np.array(vectors), outfile)
            outfile.close()
    time_dif = get_time_dif(start_time)
    print("Finished!Prepare input vectors time usage:", time_dif)
