# -*- coding: utf-8 -*-
"""
 @Time    : 2018/7/17 下午2:42
 @FileName: utils.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import itertools
import multiprocessing
import pickle
import random

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
import re
import sentencepiece as spm
np.random.seed(10245)

import json

import urllib.request


def request_qo_result(seq):
    headers = {'Content-Type': 'application/json'}
    data = {'list': seq}
    request = urllib.request.Request(url='http://10.160.40.50:1120', headers=headers,
                                     data=json.dumps(data).encode('utf-8'))
    response = urllib.request.urlopen(request, timeout=36 * 3600)
    prediction = json.loads(response.read())
    return prediction


def get_file_charset(filename):
    import chardet
    rawdata = open(filename, 'rb').read(1000)
    result = chardet.detect(rawdata)
    charenc = result['encoding']
    return charenc


def DBC2SBC(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not (0x0021 <= inside_code <= 0x7e):
            rstring += uchar
            continue
        rstring += chr(inside_code)
    return rstring


def SBC2DBC(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x0020:
            inside_code = 0x3000
        else:
            if not (0x0021 <= inside_code <= 0x7e):
                rstring += uchar
                continue
        inside_code += 0xfee0
        rstring += chr(inside_code)
    return rstring


def id_lst_to_string(id_lst, id2word):
    return ''.join([id2word[x] for x in id_lst])


def write_lst_to_file(lst, filename):
    output = '\n'.join(lst)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(output)


def dump_file(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)


def load_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_model_parameters(model):
    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            tmp = 1
            for a in parameter.size():
                tmp *= a
            total += tmp
    return total


def remove_duplciate_lst(lst):
    lst.sort()
    return list(k for k, _ in itertools.groupby(lst))


def padding(sequence, pads=0, max_len=None, dtype='int32', return_matrix_for_size=False):
    # we should judge the rank
    if True or isinstance(sequence[0], list):
        v_length = [len(x) for x in sequence]  # every sequence length
        seq_max_len = max(v_length)
        if (max_len is None) or (max_len > seq_max_len):
            max_len = seq_max_len
        v_length = list(map(lambda z: z if z <= max_len else max_len, v_length))
        x = (np.ones((len(sequence), max_len)) * pads).astype(dtype)
        for idx, s in enumerate(sequence):
            trunc = s[:max_len]
            x[idx, :len(trunc)] = trunc
        if return_matrix_for_size:
            v_matrix = np.asanyarray([map(lambda item: 1 if item < line else 0, range(max_len)) for line in v_length],
                                     dtype=dtype)
            return x, v_matrix
        return x, np.asarray(v_length, dtype='int32')
    else:
        seq_len = len(sequence)
        if max_len is None:
            max_len = seq_len
        v_vector = sequence + [0] * (max_len - seq_len)
        padded_vector = np.asarray(v_vector, dtype=dtype)
        v_index = [1] * seq_len + [0] * (max_len - seq_len)
        padded_index = np.asanyarray(v_index, dtype=dtype)
        return padded_vector, padded_index


def add2count(value, map):
    if value not in map:
        map[value] = 0
    map[value] += 1


import os


def get_dir_files(dirname):
    L = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            L.append(os.path.join(root, file))
    return L


def cleanhtmltag(raw_html):
    # cleanr = re.compile()
    cleantext = re.sub('<.*?>', '', raw_html)
    return cleantext


def clean(txt):
    txt = DBC2SBC(txt)
    txt = txt.lower()
    txt = re.sub('\s*', '', txt)
    return cleanhtmltag(txt)


def multi_process(func, lst, num_cores=multiprocessing.cpu_count(), backend='multiprocessing'):
    workers = Parallel(n_jobs=num_cores, backend=backend)
    output = workers(delayed(func)(one) for one in tqdm(lst))
    return output


def shuffle_dict(dictionary):
    keys = list(dictionary.keys())
    np.random.shuffle(keys)
    output = {}
    for one in keys:
        output[one] = dictionary[one]
    return output


def count_file(filename):
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    with open(filename, "r", encoding="utf-8", errors='ignore') as f:
        num = (sum(bl.count("\n") for bl in blocks(f)))
    return num


def lst2str(lst):
    return ' '.join(list(map(str, lst)))


def str2lst(string):
    return list(map(int, string.split()))


def reverse_map(maps):
    return {v: k for k, v in maps.items()}


def get_file_info(filename):
    with open(filename, encoding=get_file_charset(filename), errors='ignore') as f:
        for line in f:
            yield line
