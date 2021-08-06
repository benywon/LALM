# -*- coding: utf-8 -*-
"""
 @Time    : 2020/5/21 下午6:27
 @FileName: Transformer.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import warnings

import apex
import numpy as np
import torch
import torch.nn as nn
from apex.contrib.multihead_attn import EncdecMultiheadAttn
from apex.mlp import MLP
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_clones
from torch.utils.checkpoint import checkpoint

warnings.filterwarnings("ignore")
layer_norm = apex.normalization.FusedLayerNorm
gradient_checkpoint = False


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = EncdecMultiheadAttn(d_model, nhead, dropout=dropout, impl='fast')
        self.feed_forward = MLP([d_model, dim_feedforward, d_model])
        self.d_model = d_model
        self.norm1 = layer_norm(d_model)
        self.norm2 = layer_norm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=None, is_training=self.training)[0]
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)

        src2 = self.feed_forward(src2.view(-1, self.d_model)).view(src.size())
        src = src + self.dropout2(src2)

        return src


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, shared=False):
        super(TransformerEncoder, self).__init__()
        if shared:
            self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        else:
            self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask):
        output = src
        if gradient_checkpoint and self.training and mask is None:
            mask = torch.zeros((src.size(0), src.size(0)), device=src.device).byte()
        for mod in self.layers:
            if gradient_checkpoint and self.training:
                output = checkpoint(mod, output, mask)
            else:
                output = mod(output, mask)

        return output


class SelfAttention(nn.Module):
    def __init__(self, n_hidden, n_layer, n_head=6, dropout=0.1, shared=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(n_hidden, n_head, n_hidden * 4, dropout)
        self.encoder = TransformerEncoder(encoder_layer, n_layer, None, shared)
        self.output_ln = layer_norm(n_hidden)

    def forward(self, representations, mask=None):
        representations = representations.transpose(0, 1).contiguous()
        representations = self.encoder(representations, mask)
        return self.output_ln(representations.transpose(0, 1))


class Embedding(nn.Module):
    """
    the low level module of language agnostic language model
    """
    def __init__(self, vocabulary_size, n_embedding, n_hidden):
        super().__init__()
        self.n_embedding = n_embedding
        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dim=n_embedding)
        self.encoder = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden, bidirectional=False, batch_first=True)
        self.output = nn.Sequential(nn.Linear(n_hidden, n_embedding),
                                    nn.LeakyReLU(inplace=True),
                                    apex.normalization.FusedLayerNorm(n_embedding))
        self.trans = nn.Linear(n_embedding, vocabulary_size, bias=False)
        self.word_embedding.weight = self.trans.weight

    def forward(self, seq):
        word_embedding = self.word_embedding(seq)
        encoder_representations, _ = self.encoder(word_embedding)
        return encoder_representations

    def decode(self, hidden):
        hidden = self.output(hidden)
        hidden = self.trans(hidden.contiguous().view(-1, self.n_embedding))
        return hidden


class LALM(nn.Module):
    def __init__(self, vocab_size_lst, n_embedding, n_hidden, n_layer, n_head):
        super().__init__()
        self.embedding = nn.ModuleList()
        for vocab_size in vocab_size_lst:
            pad_vocab_size = (2 + vocab_size // 8) * 8
            self.embedding.append(Embedding(pad_vocab_size, n_embedding, n_hidden))
        self.n_hidden = n_hidden
        self.attention = SelfAttention(n_hidden, n_layer, n_head=n_head)

    def forward(self, inputs, language):
        if not isinstance(inputs, list):
            return self.inference(inputs, language)
        if len(inputs) == 1:
            seq = inputs[0][:, 0:-1]
            target = inputs[0][:, 1:]
        else:
            seq = inputs[0]
            target = inputs[-1]
        encoder_representations = self.embedding[language](seq)

        if len(inputs) == 1:  # uni language model
            mask = torch.triu(torch.ones((seq.size(1), seq.size(1)), device=seq.device, dtype=torch.bool),
                              diagonal=1).byte()
            hidden = self.attention(encoder_representations, mask)
        elif len(inputs) == 3:
            mask = torch.zeros((seq.size(1), seq.size(1)), device=seq.device).byte()
            index = inputs[1]
            encoder_representations = self.attention(encoder_representations, mask)
            hidden = encoder_representations.gather(1,
                                                    index.unsqueeze(2).expand(index.size(0), index.size(1),
                                                                              self.n_hidden))
        else:
            mask = inputs[1]
            target_idx = inputs[2]
            encoder_representations = self.attention(encoder_representations, mask)
            hidden = encoder_representations.masked_select(target_idx.unsqueeze(2)).reshape(-1, self.n_hidden)

        logit = self.embedding[language].decode(hidden)
        return F.cross_entropy(logit, target.contiguous().view(-1))

    @classmethod
    def get_mask_(cls, source_sentence_length, target_sentence_length):
        source_attend_matrix = np.zeros([source_sentence_length, source_sentence_length])
        target_attend_matrix = np.triu(np.ones([target_sentence_length, target_sentence_length]), 1)
        ones_matrix = np.ones([source_sentence_length, target_sentence_length])
        zeros_matrix = np.zeros([target_sentence_length, source_sentence_length])
        mask_matrix = np.block([[source_attend_matrix, ones_matrix], [zeros_matrix, target_attend_matrix]])
        return mask_matrix

    def inference(self, source, language):
        source_size = source.size(1)
        target = torch.LongTensor([[1] * source.size(0)]).view(-1, 1).cuda()
        seq = torch.cat([source, target], 1)
        predictions = []

        for i in range(20):
            encoder_representations = self.embedding[language](seq)
            mask = torch.ByteTensor(self.get_mask_(source_size, i + 1)).cuda()
            encoder_representations = self.attention(encoder_representations, mask)
            encoder_representations = encoder_representations[:, -1, :]
            prediction = F.log_softmax(self.embedding[language].decode(encoder_representations), -1)
            top = prediction.max(-1)
            target = top[1].view(1, -1)
            predictions.append(target)
            seq = torch.cat([seq, target.t()], 1)

        return torch.cat(predictions, 0).transpose(0, 1)


class LALM4QG(LALM):
    def __init__(self, vocab_size_lst, n_embedding, n_hidden, n_layer, n_head):
        super().__init__(vocab_size_lst, n_embedding, n_hidden, n_layer, n_head)

    def forward(self, inputs, language):
        if not isinstance(inputs, list):
            return self.inference(inputs, language)
        seq = inputs[0]
        target = inputs[-1]
        encoder_representations = self.embedding[language](seq)
        mask = inputs[1]
        target_idx = inputs[2]
        encoder_representations = self.attention(encoder_representations, mask)
        hidden = encoder_representations.masked_select(target_idx.unsqueeze(2)).reshape(-1, self.n_hidden)
        logit = self.embedding[language].decode(hidden)
        loss = F.cross_entropy(logit, target.contiguous().view(-1))
        return loss
