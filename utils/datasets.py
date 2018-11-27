#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-14 上午10:16
import json
import pickle
import numpy as np
from . import constants
from torch.utils.data import Dataset
from utils.util import read_data


def pad(seq, size, value):
    if len(seq) < size:
        seq.extend([value] * (size - len(seq)))
    return seq


class Vocabulary:
    """Word/char vocabulary"""

    def __init__(self):
        self.token2id = {
            constants.PAD: constants.PAD_IDX,
            constants.UNK: constants.UNK_IDX,
            constants.BOS: constants.BOS_IDX,
            constants.EOS: constants.EOS_IDX,
        }
        self.id2token = {
            constants.PAD_IDX: constants.PAD,
            constants.UNK_IDX: constants.UNK,
            constants.BOS_IDX: constants.BOS,
            constants.EOS_IDX: constants.EOS,
        }
        self.token_maxlen = -float("inf")

    def encode(self, tok):
        if tok in self.token2id:
            return self.token2id[tok]
        else:
            return constants.UNK_IDX

    def decode(self, idx):
        if id in self.id2token:
            return self.id2token[id]
        else:
            raise ValueError("No such idx: {0}".format(idx))

    def encode_seq(self, seq):
        e_seq = []
        for s in seq:
            e_seq.append(self.encode(s))
        return e_seq

    def decode_seq(self, seq):
        d_seq = []
        for i in seq:
            d_seq.append(self.decode(i))
        return d_seq

    def add_token(self, tok):
        if tok not in self.token2id:
            self.token2id[tok] = len(self.token2id)
            self.id2token[len(self.id2token)] = tok

    def save(self, path):
        with open(path, "w") as outfile:
            json.dump([self.id2token, self.token_maxlen], outfile, indent=4)
        outfile.close()

    def load(self, path):
        with open(path, 'r') as infile:
            self.id2token, self.token_maxlen = json.load(infile)
            self.id2token = {int(k): v for k, v in self.id2token.items()}
            self.token2id = {}
            for i in self.id2token.keys():
                self.token2id[self.id2token[i]] = i

    def __len__(self):
        return len(self.token2id)


class DefinitionModelingDataset(Dataset):
    def __init__(self, file, vocab_path, input_vectors_path=None, ch_vocab_path=None, hypm_path=None, use_seed=True,
                 mode='train'):
        self.data = read_data(file)
        self.voc = Vocabulary()
        self.voc.load(vocab_path)
        self.use_seed = use_seed
        self.mode = mode
        assert self.mode == "train" or "gen", "mode only in train or gen"
        if input_vectors_path is not None:
            with open(input_vectors_path, 'rb') as infile:
                self.input_vectors = pickle.load(infile)
        if ch_vocab_path is not None:
            self.ch_voc = Vocabulary()
            self.ch_voc.load(ch_vocab_path)
        if hypm_path is not None:
            with open(hypm_path, 'rb') as infile:
                self.hypm, self.hypm_weights = pickle.load(infile)
            assert len(self.hypm) == len(self.hypm_weights)

    def __getitem__(self, idx):
        inp = {
            "word": self.voc.encode(self.data[idx][0]),
            "seq": self.voc.encode_seq([constants.BOS] + self.data[idx][1] + [constants.EOS]),
            "target": self.voc.encode_seq(self.data[idx][1] + [constants.EOS] + [constants.PAD])
        }
        if hasattr(self, "ch_voc"):
            inp['chars'] = [constants.BOS_IDX] + \
                           self.ch_voc.encode_seq(list(self.data[idx][0])) + \
                           [constants.EOS_IDX]
            # CH_maxlen: +2 because EOS + BOS
            inp["CH_maxlen"] = self.ch_voc.token_maxlen + 2
        if hasattr(self, "input_vectors"):
            inp['input'] = self.input_vectors[idx]
        if hasattr(self, "hypm"):
            inp['hypm'] = self.hypm[idx]
            inp['hypm_weights'] = self.hypm_weights[idx]
        if self.use_seed:
            inp['seq'] = [self.voc.encode(self.data[idx][0])] + inp['seq']
            inp['target'] = [self.voc.encode(constants.BOS)] + inp['target']
        if self.mode == "gen":
            inp["seq"] = [inp["seq"][0]]
        return inp

    def __len__(self):
        return len(self.data)


def DefinitionModelingCollate(batch):
    batch_word = []
    batch_x = []
    batch_y = []
    is_ch = "chars" in batch[0] and "CH_maxlen" in batch[0]
    is_input = "input" in batch[0]
    is_hy = "hypm" in batch[0] and "hypm_weights" in batch[0]
    if is_ch:
        batch_ch = []
        CH_maxlen = batch[0]["CH_maxlen"]
    if is_input:
        batch_input = []
    if is_hy:
        batch_hy = []
        batch_hy_weights = []
    definition_lengths = []
    for i in range(len(batch)):
        batch_word.append(batch[i]["word"])
        batch_x.append(batch[i]["seq"])
        batch_y.append(batch[i]["target"])
        if is_ch:
            batch_ch.append(batch[i]["chars"])
        if is_input:
            batch_input.append(batch[i]["input"])
        if is_hy:
            batch_hy.append(batch[i]["hypm"])
            batch_hy_weights.append(batch[i]["hypm_weights"])
        definition_lengths.append(len(batch_x[-1]))

    definition_maxlen = max(definition_lengths)

    for i in range(len(batch)):
        batch_x[i] = pad(batch_x[i], definition_maxlen, constants.PAD_IDX)
        batch_y[i] = pad(batch_y[i], definition_maxlen, constants.PAD_IDX)
        if is_ch:
            batch_ch[i] = pad(batch_ch[i], CH_maxlen, constants.PAD_IDX)

    order = np.argsort(definition_lengths)[::-1]
    batch_x = np.array(batch_x)[order]
    batch_y = np.array(batch_y)[order]
    batch_word = np.array(batch_word)[order]
    ret_batch = {
        "word": batch_word,
        "seq": batch_x,
        "target": batch_y,
    }
    if is_ch:
        batch_ch = np.array(batch_ch)[order]
        ret_batch["chars"] = batch_ch
    if is_input:
        batch_input = np.array(batch_input)[order]
        ret_batch["input"] = batch_input
    if is_hy:
        batch_hy = np.array(batch_hy)[order]
        ret_batch["hypm"] = batch_hy
        batch_hy_weights = np.array(batch_hy_weights)[order]
        ret_batch["hypm_weights"] = batch_hy_weights
    return ret_batch
