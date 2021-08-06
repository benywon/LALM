# -*- coding: utf-8 -*-
"""
 @Time    : 2020/5/21 下午5:46
 @FileName: process_hi.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

from utils import *

sp = spm.SentencePieceProcessor()
sp.load('../data/vocab/hindi.30000.model')


def process_one(inputs):
    context, target = inputs
    context_ids = sp.encode_as_ids(context.strip())
    target_ids = sp.encode_as_ids(target.strip())
    return [context, context_ids, [[target, target_ids]]]


def process(context_file, target_file):
    context = get_file_info(context_file)
    target = get_file_info(target_file)
    data = multi_process(process_one, zip(context, target),num_cores=32)
    if 'train' in context_file:
        data = [[x[1], y[1]] for x in data for y in x[2]]
    print('have {} samples'.format(len(data)))
    return data


if __name__ == '__main__':
    train_data = process('../data/qg/hindi/answer_train.txt', '../data/qg/hindi/question_train.txt')
    dump_file(train_data, '../data/qg/train.hi.obj')
    dev_data = process('../data/qg/hindi/answer_val.txt', '../data/qg/hindi/question_val.txt')

    test_data = process('../data/qg/hindi/answer_test.txt', '../data/qg/hindi/question_test.txt')
    dump_file(dev_data+test_data, '../data/qg/dev.hi.obj')