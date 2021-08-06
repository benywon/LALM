# -*- coding: utf-8 -*-
"""
 @Time    : 2020/5/21 下午5:46
 @FileName: __init__.py.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
from utils import *


def check_parameter():
    aa = load_file('../model/model.qg.768.12.0.th')
    bb = load_file('../model/model.xunilm.768.12.1.th')
    for name in aa:
        div = aa[name] - bb[name]
        print('{}\t{}'.format(name, abs(div).max()))


def check_prediction():
    for line in get_file_info('/search/odin/bingning/program/CrossLNQG/train/output/prediction.768.3.0.txt'):
        print(line)


if __name__ == '__main__':
    train_data = [
        ('../data/train.en.obj', 0),
        ('../data/train.zh.obj', 1),
        ('../data/train.kr.obj', 2),
        ('../data/train.fr.obj', 3),
        ('../data/train.hi.obj', 7),
    ]
    for one in itertools.combinations(range(5),2):
        print([x for x in one],',')
    # check_prediction()
    # check_parameter()
