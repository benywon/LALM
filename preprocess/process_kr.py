# -*- coding: utf-8 -*-
"""
 @Time    : 2020/5/21 下午5:47
 @FileName: process_kr.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

from utils import *

sp = spm.SentencePieceProcessor()
sp.load('../data/vocab/korean.30000.model')


def one_paragraphs(paragraphs):
    one_data = []
    for paragraph in paragraphs["paragraphs"]:
        context = paragraph['context']
        doc_ids = sp.EncodeAsIds(context)
        questions = []
        for qa in paragraph["qas"]:
            question_text = qa["question"]
            question_ids = sp.EncodeAsIds(question_text)
            questions.append([question_text, question_ids])
        one_data.append([context, doc_ids, questions])
    return one_data


def process(filename):
    with open(filename, encoding=get_file_charset(filename), errors='ignore') as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    output = multi_process(one_paragraphs, dataset, num_cores=40)
    output = [y for x in output for y in x]
    if 'train' in filename:
        output = [[x[1], y[1]] for x in output for y in x[2]]
    print('{} proceed done, have {} samples'.format(filename, len(output)))
    return output


def get_krsquad():
    dev_data = process('../data/qg/korean.dev.json')
    dump_file(dev_data, '../data/qg/dev.kr.obj')
    train_data = process('../data/qg/korean.train.json')
    dump_file(train_data, '../data/qg/train.kr.obj')


if __name__ == '__main__':
    get_krsquad()
