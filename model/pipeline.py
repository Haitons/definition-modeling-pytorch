#!/usr/bin/env python3
#-*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-26 上午9:16
import torch
import numpy as np
from tqdm import tqdm
from utils import constants
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

def train_epoch(dataloader, model, optimizer, device, clip_to, logfile):
    """
    Function for training the model one epoch
        dataloader - either LanguageModeling or DefinitionModeling dataloader
        model - DefinitionModelingModel
        optimizer - optimizer to use (usually Adam)
        device - cuda/cpu
        clip_to - value to clip gradients
        logfile - where to log training
    """
    # switch model to training mode
    model.train()
    # train
    mean_batch_loss = 0
    for batch in tqdm(dataloader):
        targets = torch.t(torch.from_numpy(batch['target'])).to(device)
        # prepare model args

        data = {
            'word': torch.from_numpy(batch['word']).long().to(device),
            'seq': torch.t(torch.from_numpy(batch['seq'])).long().to(device),
        }
        if model.use_ch:
            data["chars"] = torch.from_numpy(batch['chars']).long().to(device)
        if model.use_he:
            data["hypm"] = torch.from_numpy(batch['hypm']).long().to(device)
            data["hypm_weights"] = torch.from_numpy(batch['hypm_weights']).float().to(device)

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, None, return_h=True)
        batch_loss = F.cross_entropy(
            output, targets.contiguous().view(-1),
            ignore_index=constants.PAD_IDX
        )
        optimizer.zero_grad()
        batch_loss.backward()
        clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), clip_to
        )
        optimizer.step()
        mean_batch_loss += batch_loss.item()
    mean_batch_loss = mean_batch_loss / len(dataloader)
    print(
        "Mean training loss on epoch: {0}\n".format(mean_batch_loss)
    )


def test(dataloader, model, device, logfile):
    """
    Function for testing the model on dataloader
        dataloader - either LanguageModeling or DefinitionModeling dataloader
        model - DefinitionModelingModel
        device - cuda/cpu
        logfile - where to log evaluation
    """
    # switch model to evaluation mode
    model.eval()
    # metrics
    lengths_sum = 0
    loss_sum = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            targets = torch.t(torch.from_numpy(batch['target'])).to(device)
            # prepare model args
            data = {
                'word': torch.from_numpy(batch['word']).long().to(device),
                'seq': torch.t(torch.from_numpy(batch['seq'])).long().to(device),
            }
            if model.use_ch:
                data["chars"] = torch.from_numpy(batch['chars']).long().to(device)
            if model.use_he:
                data["hypm"] = torch.from_numpy(batch['hypm']).long().to(device)
                data["hypm_weights"] = torch.from_numpy(batch['hypm_weights']).float().to(device)

            output, hidden, rnn_hs, dropped_rnn_hs = model(data, None, return_h=True)
            loss_sum += F.cross_entropy(
                output,
                targets.contiguous().view(-1),
                ignore_index=constants.PAD_IDX,
                size_average=False
            ).item()
            lengths_sum += (data["seq"] != constants.PAD_IDX).sum().item()


    perplexity = np.exp(loss_sum / lengths_sum)
    print(
        "Perplexity: {0}\n".format(perplexity)
    )
    return perplexity


