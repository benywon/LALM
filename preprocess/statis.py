# -*- coding: utf-8 -*-
"""
 @Time    : 2020/5/25 下午3:32
 @FileName: statis.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
from models.LALM import LALM
from utils import *


def get_one_name_train_dev_dataset(name='en'):
    aa = load_file('../data/train.{}.obj'.format(name))
    print(len(aa))


def get_model_size():
    zh_sp = spm.SentencePieceProcessor()
    zh_sp.load('/search/odin/bingning/data/chinese/sp.30000.model')

    en_sp = spm.SentencePieceProcessor()
    en_sp.load('/search/odin/bingning/data/english/sp.30000.model')

    ko_sp = spm.SentencePieceProcessor()
    ko_sp.load('/search/odin/bingning/data/korean/sp.30000.model')
    #
    fr_sp = spm.SentencePieceProcessor()
    fr_sp.load('/search/odin/bingning/data/french/sp.30000.model')

    hi_sp = spm.SentencePieceProcessor()
    hi_sp.load('/search/odin/bingning/data/hindi/sp.30000.model')

    sp_model = [en_sp, zh_sp, ko_sp, fr_sp, hi_sp]
    n_embedding = 1024
    n_hidden = 1024
    n_layer = 24
    n_head = 16
    vocab_size_lst = [x.get_piece_size() for x in sp_model]
    model = LALM(vocab_size_lst, n_embedding, n_hidden, n_layer, n_head)
    print('model size {}'.format(get_model_parameters(model)))

if __name__ == '__main__':
    get_model_size()
    # get_one_name_train_dev_dataset('en')
