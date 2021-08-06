# -*- coding: utf-8 -*-
"""
 @Time    : 2020/6/2 下午5:52
 @FileName: model.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

from utils import *
import math
import warnings

import apex
import torch
import torch.nn as nn
from apex.contrib.multihead_attn import SelfMultiheadAttn, EncdecMultiheadAttn
from torch.nn import functional as F


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layer_size):
        super().__init__()
        self.vocab_size = vocab_size
