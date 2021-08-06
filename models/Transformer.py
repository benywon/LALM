# -*- coding: utf-8 -*-
"""
 @Time    : 2020/4/7 下午12:22
 @FileName: Transformer.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import math
import warnings

import apex
import torch
import torch.nn as nn
from apex.contrib.multihead_attn import SelfMultiheadAttn, EncdecMultiheadAttn
from torch.nn import functional as F

warnings.filterwarnings("ignore")

LayerNorm = apex.normalization.FusedLayerNorm


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn = SelfMultiheadAttn(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn = SelfMultiheadAttn(d_model, nhead, dropout=dropout)
        self.multihead_attn = EncdecMultiheadAttn(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.gelu


class Transformer(nn.Transformer):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", custom_encoder=None, custom_decoder=None):
        super().__init__(d_model, nhead, num_encoder_layers,
                         num_decoder_layers, dim_feedforward, dropout,
                         activation, custom_encoder, custom_decoder)
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def inference(self, memory, tgt, tgt_mask):
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return output[-1, :]


class PosEmbedding(nn.Module):
    def __init__(self, ntoken, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.word_embedding = nn.Embedding(ntoken, d_model)
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.word_embedding(x)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EmbeddingEncoder(nn.Module):
    def __init__(self, ntoken, n_embedding, n_hidden):
        super().__init__()
        self.word_embedding = nn.Embedding(ntoken, n_embedding)
        self.encoder = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden // 2, bidirectional=True)

    def forward(self, x):
        x = self.word_embedding(x)
        x, _ = self.encoder(x)
        return x


class EmbeddingDecoder(nn.Module):
    def __init__(self, ntoken, n_embedding, n_hidden):
        super().__init__()
        self.word_embedding = nn.Embedding(ntoken, n_embedding)
        self.encoder = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden)
        self.output = nn.Sequential(nn.Linear(n_hidden, n_embedding),
                                    nn.LeakyReLU(inplace=True),
                                    apex.normalization.FusedLayerNorm(n_embedding))
        self.trans = nn.Linear(n_embedding, ntoken, bias=False)
        self.word_embedding.weight = self.trans.weight
        self.n_embedding = n_embedding

    def forward(self, x):
        x = self.word_embedding(x)
        x, _ = self.encoder(x)
        return x

    def inference(self, x, hidden):
        x = self.word_embedding(x)
        x, hidden = self.encoder(x, hidden)
        return x, hidden

    def decode(self, hidden):
        hidden = self.output(hidden)
        hidden = self.trans(hidden.contiguous().view(-1, self.n_embedding))
        return hidden


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, n_embedding, n_hidden):
        super().__init__()
        self.vocab_size = vocabulary_size
        self.n_embedding = n_embedding
        self.encoder = EmbeddingEncoder(vocabulary_size, n_embedding, n_hidden)
        self.decoder = EmbeddingDecoder(vocabulary_size, n_embedding, n_hidden)
        self.encoder.word_embedding = self.decoder.word_embedding

    def forward(self, seq):
        return self.encoder(seq)

    def encode(self, seq):
        return self.decoder(seq)

    def decode(self, hidden):
        return self.decoder.decode(hidden).view(-1, self.vocab_size)


class TransformerX(nn.Module):
    def __init__(self, vocab_size_lst, nemb, nhid, nlayers, nhead, dropout=0.1):
        super().__init__()
        self.embedding = nn.ModuleList()
        for vocab_size in vocab_size_lst:
            pad_vocab_size = (2 + vocab_size // 8) * 8
            self.embedding.append(Embedding(pad_vocab_size, nemb, nhid))
        self.model = Transformer(nhid, nhead, nlayers, nlayers, nhid * 4, dropout=dropout)

    def forward(self, inputs, lang):
        [source, target] = inputs
        if target is None:
            return self.inference(source)
        source = source.transpose(0, 1)
        target = target.transpose(0, 1)
        source_embedding = self.embedding[lang](source)
        target_source = target[0:-1, :]
        target_target = target[1:, :]
        target_embedding = self.embedding[lang].encode(target_source)
        len_s = target_target.size(0)
        mask = torch.triu(torch.ones((len_s, len_s), device=source.device, dtype=torch.bool), diagonal=1).byte()
        target_output = self.model(source_embedding, target_embedding, None, mask)
        logit = self.embedding[lang].decode(target_output)
        return F.cross_entropy(logit, target_target.contiguous().view(-1))

    def inference(self, source):
        source = source.transpose(0, 1)
        source_embedding = self.encoder_embedding(source)
        memory = self.model.encoder(source_embedding)
        target = torch.LongTensor([[1] * source.size(1)]).cuda()
        target_embedding, rnn_hidden = self.decoder_embedding.inference(target, None)
        predictions = []
        probabilities = []
        for i in range(20):
            len_s = i + 1
            mask = torch.triu(torch.ones((len_s, len_s), device=source.device, dtype=torch.bool), diagonal=1).byte()
            output = self.model.inference(memory, target_embedding, mask)
            prediction = F.log_softmax(self.output(output), -1)
            top = prediction.max(-1)
            scores = top[0].view(1, -1)
            target = top[1].view(1, -1)
            predictions.append(target)
            probabilities.append(scores)
            current_embedding, rnn_hidden = self.decoder_embedding.inference(target, rnn_hidden)
            target_embedding = torch.cat([target_embedding, current_embedding], 0)
        return torch.cat(predictions, 0).transpose(0, 1)
