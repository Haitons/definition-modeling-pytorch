#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-13 下午1:35
import os
import numpy as np
import torch
import time
import argparse
from torch import nn
import json
from tqdm import tqdm
from model.model import RNNModel
# from model.pipeline import train_epoch, test
from utils.datasets import DefinitionModelingDataset, DefinitionModelingCollate
from utils import constants
from utils.util import get_time_dif
from torch.utils.data import DataLoader

# Read all arguments and prepare all stuff for training

parser = argparse.ArgumentParser(description='Pytorch Definition Sequence Model.')
# Common data arguments
parser.add_argument(
    "--voc", type=str, required=True, help="location of vocabulary file"
)
# Definitions data arguments
parser.add_argument(
    '--train_defs', type=str, required=True,
    help="location of txt file with train definitions."
)
parser.add_argument(
    '--eval_defs', type=str, required=True,
    help="location of txt file with metrics definitions."
)
parser.add_argument(
    '--test_defs', type=str, required=True,
    help="location of txt file with test definitions"
)
parser.add_argument(
    '--input_train', type=str, required=False,
    help="location of train vectors for Input conditioning"
)
parser.add_argument(
    '--input_eval', type=str, required=False,
    help="location of metrics vectors for Input conditioning"
)
parser.add_argument(
    '--input_test', type=str, required=False,
    help="location of test vectors for Input conditioning"
)
parser.add_argument(
    '--hypm_train', type=str, required=False,
    help="location of train hypernyms for Hypernyms conditioning"
)
parser.add_argument(
    '--hypm_eval', type=str, required=False,
    help="location of metrics hypernyms for Hypernyms conditioning"
)
parser.add_argument(
    '--hypm_test', type=str, required=False,
    help="location of test hypernyms for Hypernyms conditioning"
)
parser.add_argument(
    '--ch_voc', type=str, required=False,
    help="location of CH vocabulary file"
)
# Model parameters arguments
parser.add_argument(
    '--rnn_type', type=str, default='GRU',
    help='type of recurrent neural network(LSTM,GRU)'
)
parser.add_argument(
    '--emdim', type=int, default=300,
    help='size of word embeddings'
)
parser.add_argument(
    '--hidim', type=int, default=300,
    help='numbers of hidden units per layer'
)
parser.add_argument(
    '--nlayers', type=int, default=2,
    help='number of recurrent neural network layers'
)
parser.add_argument(
    '--use_seed', action='store_true',
    help='whether to use Seed conditioning or not'
)
parser.add_argument(
    '--use_input', action='store_true',
    help='whether to use Input conditioning or not'
)
parser.add_argument(
    '--use_hidden', action='store_true',
    help='whether to use Hidden conditioning or not'
)
parser.add_argument(
    '--use_gated', action='store_true',
    help='whether to use Gated conditioning or not'
)
parser.add_argument(
    '--use_ch', action='store_true',
    help='use character level CNN'
)
parser.add_argument(
    '--ch_emb_size', type=int, required=False,
    help="size of embeddings in CH conditioning"
)
parser.add_argument(
    '--ch_feature_maps', type=int, required=False, nargs="+",
    help="list of feature map sizes in CH conditioning"
)
parser.add_argument(
    '--ch_kernel_sizes', type=int, required=False, nargs="+",
    help="list of kernel sizes in CH conditioning"
)
parser.add_argument(
    '--use_he', action='store_true',
    help='use hypernym embeddings'
)
# Training and dropout arguments
parser.add_argument(
    '--lr', type=int, default=0.001,
    help='initial learning rate'
)
parser.add_argument(
    "--decay_factor", type=float, default=0.1,
    help="factor to decay lr"
)
parser.add_argument(
    '--decay_patience', type=int, default=0,
    help="after number of patience epochs - decay lr"
)
parser.add_argument(
    '--clip', type=int, default=5,
    help='value to clip norm of gradients to'
)
parser.add_argument(
    '--epochs', type=int, default=40,
    help='upper epoch limit'
)
parser.add_argument(
    '--batch_size', type=int, default=15,
    help='batch size'
)
parser.add_argument(
    '--tied', action='store_true',
    help='tie the word embedding and softmax weights'
)
parser.add_argument(
    '--random_seed', type=int, default=22222,
    help='random seed'
)
parser.add_argument(
    '--dropout', type=float, default=0,
    help='dropout applied to layers (0 = no dropout)'
)
parser.add_argument(
    '--dropouth', type=float, default=0,
    help='dropout for rnn layers (0 = no dropout)'
)
parser.add_argument(
    '--dropouti', type=float, default=0,
    help='dropout for input embedding layers (0 = no dropout)'
)
parser.add_argument(
    '--dropoute', type=float, default=0,
    help='dropout to remove words from embedding layer (0 = no dropout)'
)
parser.add_argument(
    '--wdrop', type=float, default=0,
    help='amount of weight dropout to apply to the RNN hidden to hidden matrix'
)
parser.add_argument(
    '--wdecay', type=float, default=1.2e-6,
    help='weight decay applied to all weights'
)
parser.add_argument(
    '--alpha', type=float, default=2,
    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)'
)
parser.add_argument(
    '--beta', type=float, default=1,
    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)'
)
# Utility arguments
parser.add_argument(
    "--exp_dir", type=str, required=True,
    help="where to save all stuff about training"
)
parser.add_argument(
    "--w2v_weights", type=str, required=False,
    help="path to pretrained embeddings to init"
)
parser.add_argument(
    "--fix_embeddings", action="store_true",
    help="whether to update embedding matrix or not"
)
parser.add_argument(
    '--cuda', action='store_true',
    help='use CUDA'
)
# set default args
parser.set_defaults(tied=False, use_ch=False, use_he=False, fix_embeddings=True,
                    use_seed=False, use_input=False, use_hidden=False, use_gated=True)
# read args
args = vars(parser.parse_args())

train_dataset = DefinitionModelingDataset(
    file=args["train_defs"],
    vocab_path=args["voc"],
    input_vectors_path=args["input_train"],
    ch_vocab_path=args["ch_voc"],
    use_seed=args["use_seed"],
    hypm_path=args["hypm_train"],
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args["batch_size"],
    collate_fn=DefinitionModelingCollate,
    shuffle=True,
    num_workers=2
)
valid_dataset = DefinitionModelingDataset(
    file=args["eval_defs"],
    vocab_path=args["voc"],
    input_vectors_path=args["input_eval"],
    ch_vocab_path=args["ch_voc"],
    use_seed=args["use_seed"],
    hypm_path=args["hypm_eval"],
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=args["batch_size"],
    collate_fn=DefinitionModelingCollate,
    shuffle=True,
    num_workers=2
)

if args["use_input"] or args["use_hidden"] or args["use_gated"]:
    assert args["input_train"] is not None, ("--use_input or "
                                             "--use_hidden or "
                                             "--use_gated is used "
                                             "--input_train is required")
    assert args["input_eval"] is not None, ("--use_input or "
                                            "--use_hidden or "
                                            "--use_gated is used "
                                            "--input_eval is required")
    assert args["input_test"] is not None, ("--use_input or "
                                            "--use_hidden or "
                                            "--use_gated is used "
                                            "--input_test is required")
    args["input_dim"] = train_dataset.input_vectors.shape[1]
if args["use_ch"]:
    assert args["ch_voc"] is not None, ("--ch_voc is required "
                                        "if --use_ch")
    assert args["ch_emb_size"] is not None, ("--ch_emb_size is required "
                                             "if --use_ch")
    assert args["ch_feature_maps"] is not None, ("--ch_feature_maps is "
                                                 "required if --use_ch")
    assert args["ch_kernel_sizes"] is not None, ("--ch_kernel_sizes is "
                                                 "required if --use_ch")

    args["n_ch_tokens"] = len(train_dataset.ch_voc.token2id)
    args["ch_maxlen"] = train_dataset.ch_voc.token_maxlen + 2
if args["use_he"]:
    assert args["hypm_train"] is not None, ("--use_he is used "
                                            "--hypm_train is required")
    assert args["hypm_eval"] is not None, ("--use_he is used "
                                           "--hypm_eval is required")
    assert args["hypm_test"] is not None, ("--use_he is used "
                                           "--hypm_test is required")

args["vocab_size"] = len(train_dataset.voc.token2id)
# Set the random seed manually for reproducibility
np.random.seed(args["random_seed"])
torch.manual_seed(args["random_seed"])
if torch.cuda.is_available():
    if not args["cuda"]:
        print('WARNING:You have a CUDA device,so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed(args["random_seed"])
device = torch.device('cuda' if args["cuda"] else 'cpu')


def train():
    print('=========model architecture==========')
    model = RNNModel(args).to(device)
    print(model)
    print('=============== end =================')
    loss_fn = nn.CrossEntropyLoss(ignore_index=constants.PAD_IDX)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, args["lr"], weight_decay=args["wdecay"])
    print('Training and evaluating...')
    start_time = time.time()
    if not os.path.exists(args["exp_dir"]):
        os.makedirs(args["exp_dir"])
    best_ppl = 9999999
    last_improved = 0
    require_improvement = 5
    with open(args["exp_dir"] + "params.json", "w") as outfile:
        json.dump(args, outfile, indent=4)
    for epoch in range(args["epochs"]):
        model.training = True
        loss_epoch = []
        for batch, inp in enumerate(tqdm(train_dataloader, desc='Epoch: %03d' % (epoch + 1), leave=False)):
            data = {
                'word': torch.from_numpy(inp['word']).long().to(device),
                'seq': torch.t(torch.from_numpy(inp['seq'])).long().to(device),
            }
            if model.use_input:
                data["input_vectors"] = torch.from_numpy(inp['input']).to(device)
            if model.use_ch:
                data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
            if model.use_he:
                data["hypm"] = torch.from_numpy(inp['hypm']).long().to(device)
                data["hypm_weights"] = torch.from_numpy(inp['hypm_weights']).float().to(device)
            targets = torch.t(torch.from_numpy(inp['target'])).to(device)
            output, hidden, rnn_hs, dropped_rnn_hs = model(data, None, return_h=True)
            loss = loss_fn(output, targets.view(-1))
            optimizer.zero_grad()
            # Activiation Regularization
            # if args.alpha:
            #     loss = loss + sum(
            #         args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # if args.beta:
            #     loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()
            # `clip_grad_norm`
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["clip"])
            optimizer.step()
            loss_epoch.append(loss.item())
        train_loss = np.mean(loss_epoch)
        train_ppl = np.exp(train_loss)
        valid_loss, valid_ppl = evaluate(model, valid_dataloader, device)

        if valid_ppl < best_ppl:
            best_ppl = valid_ppl
            last_improved = epoch
            torch.save(model.state_dict(), args["exp_dir"] +
                       'defseq_model_params_%s_min_ppl.pkl' % (epoch + 1)
                       )
            improved_str = '*'
        else:
            improved_str = ''
        time_dif = get_time_dif(start_time)
        msg = 'Epoch: {0:>6},Train Loss: {1:>6.6}, Train Ppl: {2:>6.6},' \
              + ' Val loss: {3:>6.6}, Val Ppl: {4:>6.6},Time:{5} {6}'
        print(msg.format(epoch + 1, train_loss, train_ppl, valid_loss, valid_ppl, time_dif, improved_str))
        if epoch - last_improved > require_improvement:
            print("No optimization for a long time, auto-stopping...")
            break
    return 1


def evaluate(model, dataloader, device='cpu'):
    model.training = False
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        total_loss = []
        for inp in dataloader:
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


if __name__ == "__main__":
    print('=============user config=============')
    for key, value in sorted(args.items()):
        print('{key}:{value}'.format(key=key, value=value))
    print('=============== end =================')
    train()
