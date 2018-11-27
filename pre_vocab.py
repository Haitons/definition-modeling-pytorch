#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-14 下午1:34
import time
import argparse
from utils.datasets import Vocabulary
from utils.util import get_time_dif, read_data, read_hypernyms

parser = argparse.ArgumentParser(description="Prepare vocabulary for model")
parser.add_argument(
    '--defs', type=str, required=True, nargs="+",
    help="location of train,valid,test definitions file."
)
parser.add_argument(
    '--hypm', type=str, required=False,
    help="location of bag of hypernyms."
)
parser.add_argument(
    '--save', type=str,default="../data/processed/vocab.json",
    help="where to save word vocabulary."
)
parser.add_argument(
    '--save_chars', type=str,default="../data/processed/char_vocab.json",
    help="where to save char vocabulary."
)
args = parser.parse_args()
if len(args.defs) != 3:
    parser.error("--defs must have both train,valid and test definitions file.")

if __name__ == "__main__":
    voc = Vocabulary()
    char_voc = Vocabulary()
    start_time = time.time()
    print("Start build the vocabulary at {}".format(time.asctime(time.localtime(start_time))))
    for filepath in args.defs:
        data = read_data(filepath)
        for elem in data:
            voc.add_token(elem[0])
            char_voc.token_maxlen = max(len(elem[0]), char_voc.token_maxlen)
            for c in elem[0]:
                char_voc.add_token(c)
            definition = elem[1]
            for d in definition:
                voc.add_token(d)
    if args.hypm is not None:
        _, hypm_token = read_hypernyms(args.hypm)
        for h in hypm_token:
            voc.add_token(h)
    voc.save(args.save)
    char_voc.save(args.save_chars)
    time_dif = get_time_dif(start_time)
    print("Finished! Build vocabulary time usage:", time_dif)
