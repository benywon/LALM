# -*- coding: utf-8 -*-
"""
 @Time    : 2020/5/21 下午5:46
 @FileName: process_zh.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

from utils import *
import json

sp = spm.SentencePieceProcessor()
sp.load('../data/vocab/chinese.30000.model')

def one_paragraphs(paragraphs):
    one_data = []
    context = '{}##{}'.format(paragraphs['title'], paragraphs['content'])
    doc_ids = sp.EncodeAsIds(context)
    questions = []
    for question_text in paragraphs["questions"]:
        question_ids = sp.EncodeAsIds(question_text)
        questions.append([question_text, question_ids])
    one_data.append([context, doc_ids, questions])
    return one_data


def process(filename):
    with open(filename, encoding=get_file_charset(filename), errors='ignore') as dataset_file:
        dataset = json.load(dataset_file)
    output = multi_process(one_paragraphs, dataset, num_cores=40)
    output = [y for x in output for y in x]
    if 'train' in filename:
        output = [[x[1], y[1]] for x in output for y in x[2]]
    print('{} proceed done, have {} samples'.format(filename, len(output)))
    return output


def get_lab():
    dev_data = process('../data/qg/chinese.dev.json')
    dump_file(dev_data, '../data/qg/dev.zh.obj')
    test_data = process('../data/qg/chinese.test.json')
    dump_file(test_data, '../data/qg/test.zh.obj')
    train_data = process('../data/qg/chinese.train.json')
    dump_file(train_data, '../data/qg/train.zh.obj')
