#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-20 下午2:41
import torch
from torch import nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    """
    Class for CH conditioning
    """

    def __init__(self, n_ch_tokens, ch_maxlen, ch_emb_size,
                 ch_feature_maps, ch_kernel_sizes, ch_drop=0.25):
        super(CharCNN, self).__init__()
        assert len(ch_feature_maps) == len(ch_kernel_sizes)

        self.n_ch_tokens = n_ch_tokens
        self.ch_maxlen = ch_maxlen
        self.ch_emb_size = ch_emb_size
        self.ch_feature_maps = ch_feature_maps
        self.ch_kernel_sizes = ch_kernel_sizes
        self.ch_drop = nn.Dropout(ch_drop)

        self.feature_mappers = nn.ModuleList()
        for i in range(len(self.ch_feature_maps)):
            reduced_length = self.ch_maxlen - self.ch_kernel_sizes[i] + 1
            self.feature_mappers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=self.ch_feature_maps[i],
                        kernel_size=(
                            self.ch_kernel_sizes[i],
                            self.ch_emb_size
                        )
                    ),
                    nn.Tanh(),
                    nn.MaxPool2d(kernel_size=(reduced_length, 1))
                )
            )

        self.embs = nn.Embedding(
            self.n_ch_tokens,
            self.ch_emb_size,
            padding_idx=0
        )

    def forward(self, x):
        # x - [batch_size x maxlen]
        bsize, length = x.size()
        assert length == self.ch_maxlen
        x_embs = self.embs(x).view(bsize, 1, self.ch_maxlen, self.ch_emb_size)

        cnn_features = []
        for i in range(len(self.ch_feature_maps)):
            cnn_features.append(
                self.feature_mappers[i](x_embs).view(bsize, -1)
            )

        return self.ch_drop(torch.cat(cnn_features, dim=1))

    def init_ch(self):
        initrange = 0.5 / self.ch_emb_size
        with torch.no_grad():
            nn.init.uniform_(self.embs.weight, -initrange, initrange)
            for name, p in self.feature_mappers.named_parameters():
                if "bias" in name:
                    nn.init.constant_(p, 0)
                elif "weight" in name:
                    nn.init.xavier_uniform_(p)


class Hypernym(nn.Module):
    """
        Class for HE conditioning
    """

    def __init__(self, embedding_dim, embeddings, device):
        super(Hypernym, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = embeddings
        self.device = device

    def forward(self, inputs):
        batch_hynm = inputs[0]
        batch_hynm_weights = inputs[1]
        batch_sum = []
        for hynm, weights in zip(batch_hynm, batch_hynm_weights):
            weighted_sum = torch.zeros(self.embedding_dim).to(self.device)
            for h, w in zip(hynm, weights):
                word_emb = self.embeddings(h)
                weighted_sum += w * word_emb
            batch_sum.append(torch.unsqueeze(weighted_sum, 0))
        return torch.cat(batch_sum, dim=0)


class Hidden(nn.Module):
    """
    Class for Hidden conditioning
    """

    def __init__(self, in_size, out_size):
        super(Hidden, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(
            in_features=self.in_size,
            out_features=self.out_size
        )
        self.h_bn = nn.BatchNorm1d(self.out_size)

    def forward(self, rnn_type, num_layers, v, hidden):
        batch_size = v.size(0)
        if rnn_type == 'LSTM':
            inp_h = torch.cat(
                [torch.unsqueeze(v, 0).expand(num_layers, batch_size, -1).contiguous(),
                 hidden[0]], dim=-1)
            hidden = F.tanh(self.linear(inp_h))
        else:
            inp_h = torch.cat(
                [torch.unsqueeze(v, 0).expand(num_layers, batch_size, -1).contiguous(),
                 hidden], dim=-1)
            hidden = F.tanh(self.linear(inp_h))
        return hidden

    def init_hidden(self):
        # with torch.no_grad():
        #     nn.init.xavier_uniform_(self.linear.weight)
        #     nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.linear.bias, 0.0)
        nn.init.xavier_normal_(self.linear.weight)


class Gated(nn.Module):
    """
    Class for Gated conditioning
    """

    def __init__(self, cond_size, hidden_size):
        super(Gated, self).__init__()
        self.cond_size = cond_size
        self.hidden_size = hidden_size
        self.in_size = self.cond_size + self.hidden_size
        self.zt_linear = nn.Linear(
            in_features=self.in_size,
            out_features=self.hidden_size
        )
        self.zt_bn = nn.BatchNorm1d(self.hidden_size)
        self.rt_linear = nn.Linear(
            in_features=self.in_size,
            out_features=self.cond_size
        )
        self.rt_bn = nn.BatchNorm1d(self.cond_size)
        self.ht_linear = nn.Linear(
            in_features=self.in_size,
            out_features=self.hidden_size
        )
        self.ht_bn = nn.BatchNorm1d(self.hidden_size)

    def forward(self, rnn_type, num_layers, v, hidden):
        batch_size = v.size(0)
        if rnn_type == 'LSTM':
            inp_h = torch.cat(
                [torch.unsqueeze(v, 0).expand(num_layers, batch_size, -1).contiguous(),
                 hidden[0]], dim=-1)
            z_t = F.sigmoid(self.zt_linear(inp_h))
            r_t = F.sigmoid(self.rt_linear(inp_h))
            mul = torch.mul(r_t, v)
            hidden_ = torch.cat([mul, hidden[0]], dim=-1)
            hidden_ = F.tanh(self.ht_linear(hidden_))
            hidden = (torch.mul((1 - z_t), hidden[0]) + torch.mul(z_t, hidden_), hidden[1])
        else:
            inp_h = torch.cat(
                [torch.unsqueeze(v, 0).expand(num_layers, batch_size, -1).contiguous(),
                 hidden],
                dim=-1)
            z_t = F.sigmoid(self.zt_linear(inp_h))
            r_t = F.sigmoid(self.rt_linear(inp_h))
            mul = torch.mul(r_t, v)
            hidden_ = torch.cat([mul, hidden], dim=-1)
            hidden_ = F.tanh(self.ht_linear(hidden_))
            hidden = torch.mul((1 - z_t), hidden) + torch.mul(z_t, hidden_)
        return hidden

    def init_gated(self):
        # with torch.no_grad():
        #     nn.init.xavier_uniform_(self.linear1.weight)
        #     nn.init.xavier_uniform_(self.linear2.weight)
        #     nn.init.xavier_uniform_(self.linear3.weight)
        #     nn.init.constant_(self.linear1.bias, 0)
        #     nn.init.constant_(self.linear2.bias, 0)
        #     nn.init.constant_(self.linear3.bias, 0)
        nn.init.constant_(self.zt_linear.bias, 0.0)
        nn.init.xavier_normal_(self.zt_linear.weight)
        nn.init.constant_(self.rt_linear.bias, 0.0)
        nn.init.xavier_normal_(self.rt_linear.weight)
        nn.init.constant_(self.ht_linear.bias, 0.0)
        nn.init.xavier_normal_(self.ht_linear.weight)
