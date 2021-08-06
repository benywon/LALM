# -*- coding: utf-8 -*-
"""
 @Time    : 2020/5/26 上午11:42
 @FileName: test.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import argparse
import math
import sys
import time

import apex
import nltk
import os

sys.path.append("..")
from QGevaluation.eval import COCOEvalCap
from models.LALM import LALM4QG

from utils import *
import sentencepiece as spm

'''
python3 -m torch.distributed.launch --nproc_per_node=8 train_sentence_piece_selection.py 
'''
from apex import amp

zh_sp = spm.SentencePieceProcessor()
zh_sp.load('../data/vocab/chinese.30000.model')

en_sp = spm.SentencePieceProcessor()
en_sp.load('../data/vocab/english.30000.model')

ko_sp = spm.SentencePieceProcessor()
ko_sp.load('../data/vocab/korean.30000.model')
#
fr_sp = spm.SentencePieceProcessor()
fr_sp.load('../data/vocab/french.30000.model')

hi_sp = spm.SentencePieceProcessor()
hi_sp.load('../data/vocab/hindi.30000.model')

bu_sp = spm.SentencePieceProcessor()
bu_sp.load('../data/vocab/burmese.30000.model')

de_sp = spm.SentencePieceProcessor()
de_sp.load('../data/vocab/german.30000.model')

vi_sp = spm.SentencePieceProcessor()
vi_sp.load('../data/vocab/vietnam.30000.model')

ja_sp = spm.SentencePieceProcessor()
ja_sp.load('../data/vocab/japanese.30000.model')

mi_sp = spm.SentencePieceProcessor()
mi_sp.load('../data/vocab/minnan.30000.model')

sp_model = [en_sp, zh_sp, ko_sp, fr_sp, hi_sp, bu_sp, de_sp, vi_sp, ja_sp, mi_sp]

vocab_size_lst = [x.get_piece_size() for x in sp_model]

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=int, default=4)
args = parser.parse_args()

data_number = args.data
os.environ["CUDA_VISIBLE_DEVICES"] = str(data_number % 4)

data_map = {0: 'en',
            1: 'zh',
            2: 'kr',
            3: 'fr',
            4: 'hi'}

n_embedding = 128
n_hidden = 768
n_layer = 12
n_head = 12
batch_size = 16
doc_max_length_size = 256 + 256
question_max_length_size = 20

vocab_size = 30000

pair_name = 'all'


def load_model(epoch):
    model = LALM4QG(vocab_size_lst, n_embedding, n_hidden, n_layer, n_head)
    pre_train = True
    if pre_train:
        state_dict = load_file(
            '../model/model.qg.{}.{}.{}.th'.format(n_hidden, pair_name, epoch))
        for name, para in model.named_parameters():
            if name not in state_dict:
                print('unload parameter {}'.format(name))
                continue
            para.data = torch.FloatTensor(state_dict[name])
    print('model size {}'.format(get_model_parameters(model)))
    model.cuda()
    model.eval()
    [model] = amp.initialize([model], opt_level='O2', verbosity=0)
    return model


def clean_prediction(prediction):
    if 2 in prediction:
        end = prediction.index(2)
    elif 0 in prediction:
        end = prediction.index(0)
    else:
        end = len(prediction)
    prediction = prediction[0:end]
    return [x for x in prediction if x < vocab_size]


def process_output(output):
    output = [clean_prediction(x) for x in output]
    output = [sp.decode_ids(x) for x in output]
    return output


def transform_predictions(tokens):
    if language == 0:
        lst = [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens, 'english')]
        return ' '.join(lst).lower()
    elif language == 3:
        lst = [token for token in nltk.word_tokenize(tokens, 'french')]
        return ' '.join(lst).lower()
    elif language == 1:
        return ' '.join(list(tokens.strip()))
    return tokens.strip()


def evaluation(truths, predictions, docs, epo):
    predictions = [transform_predictions(x) for x in predictions]
    truths = [[transform_predictions(y) for y in x] for x in truths]
    truths = [x if len(x) > 0 else [''] for x in truths]
    eval = COCOEvalCap(reference=truths, prediction=predictions, only_bleu=False if language != 2 else True)
    result = eval.evaluate()
    output = []
    # for d, p, r in zip(docs, predictions, truths):
    #     output.append(d)
    #     output.append(p)
    #     output.append('\t'.join([x for x in r]))
    #     output.append('****' * 30)
    # write_lst_to_file(output, 'output/prediction.{}.{}.{}.txt'.format(n_hidden, primary, secondary, data_name, epo))
    return result


def dev(dev_data, model, epo):
    model.eval()
    total = len(dev_data)
    predictions = []
    truths = []
    docs = []
    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size)):
            sample = dev_data[i:i + batch_size]
            context_raw = [x[1][::-1] for x in sample]
            question = [[y[0] for y in x[2]] for x in sample]
            context, _ = padding(context_raw, max_len=512)
            context = torch.LongTensor(np.flip(context, 1).copy()).cuda(non_blocking=True)
            output = model(context, language)
            output = output.cpu().data.numpy().tolist()
            predictions.extend(process_output(output))
            docs.extend([sp.decode_ids([y for y in x if y < vocab_size]) for x in context_raw])
            truths.extend([x for x in question])
    try:
        return evaluation(truths, predictions, docs, epo)
    except:
        return ['wrong']


if __name__ == '__main__':
    res = []
    data_name = data_map[data_number]
    dev_data = load_file('../data/qg/dev.{}.obj'.format(data_name))
    language = data_number
    sp = sp_model[language]
    vocab_size = sp.get_piece_size()
    res.append(data_name)
    res.append('dev data size is {}'.format(len(dev_data)))
    for epoch in range(0, 5):
        res.append('====={}====='.format(epoch))
        model = load_model(epoch)
        res.extend(dev(dev_data, model, epoch))
        print(res)
        write_lst_to_file(res, 'result/{}.{}.txt'.format(pair_name, data_name))
