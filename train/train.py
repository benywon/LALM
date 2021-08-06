# -*- coding: utf-8 -*-
"""
 @Time    : 2020/5/24 上午10:55
 @FileName: train.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import argparse
import sys
import time

import apex

sys.path.append("..")
from models.LALM import LALM4QG

from utils import *
import torch.distributed as dist
import sentencepiece as spm

'''
python3 -m torch.distributed.launch --nproc_per_node=8 train_sentence_piece_selection.py 
'''
from apex import amp

torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--num", type=int, default=0)
args = parser.parse_args()
lan_num = args.num
torch.cuda.set_device(args.local_rank)

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

train_data = [
    ('../data/qg/train.en.obj', 0),
    ('../data/qg/train.zh.obj', 1),
    ('../data/qg/train.kr.obj', 2),
    ('../data/qg/train.fr.obj', 3),
    ('../data/qg/train.hi.obj', 4),
]

combinations = [[0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 3],
                [2, 4],
                [3, 4]]
[primary, secondary] = combinations[lan_num]

n_embedding = 128
n_hidden = 768
n_layer = 12
n_head = 12
batch_size = 16
question_max_length_size = 20
doc_max_length_size = 512-question_max_length_size
model = LALM4QG(vocab_size_lst, n_embedding, n_hidden, n_layer, n_head)
pre_train = True
epoch = 6
if pre_train:
    state_dict = load_file('../model/model.xunilm.{}.{}.{}.th'.format(n_hidden, n_layer, epoch))
    for name, para in model.named_parameters():
        if name not in state_dict:
            print('unload parameter {}'.format(name))
            continue
        para.data = torch.FloatTensor(state_dict[name])
print('model size {}'.format(get_model_parameters(model)))
model.cuda()
max_learning_rate = 4e-5
warm_up_steps = 100
iters_to_accumulate = 1
log_interval = 100
optim = apex.optimizers.FusedAdam(
    # model.parameters(),
    [
        {'params': model.attention.parameters()},
        {'params': model.embedding.parameters(), 'lr': 0}
    ]
    ,
    weight_decay=1.0e-2,
    # max_grad_norm=0.5,
    lr=max_learning_rate
)

model, optim = amp.initialize(model, optim, opt_level="O2", verbosity=0)
model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

update_number = 0


def get_shuffle_data(data):
    pool = {}
    for one in data:
        length = len(one[-1] + one[-2]) // 4
        if length not in pool:
            pool[length] = []
        pool[length].append(one)
    for one in pool:
        np.random.shuffle(pool[one])
    length_lst = list(pool.keys())
    np.random.shuffle(length_lst)
    whole_data = [x for y in length_lst for x in pool[y]]
    remove_data_size = len(whole_data) % dist.get_world_size()
    thread_data = [whole_data[x + args.local_rank] for x in
                   range(0, len(whole_data) - remove_data_size, dist.get_world_size())]
    return thread_data


def metric_sum(val):
    tensor = torch.tensor(val).cuda()
    dist.reduce(tensor, 0)
    return tensor.item()


def get_partial_attention_matrix(source_sentence_length, target_sentence_length):
    source_attend_matrix = np.zeros([source_sentence_length, source_sentence_length])
    target_attend_matrix = np.triu(np.ones([target_sentence_length, target_sentence_length]), 1)
    ones_matrix = np.ones([source_sentence_length, target_sentence_length])
    zeros_matrix = np.zeros([target_sentence_length, source_sentence_length])
    mask_matrix = np.block([[source_attend_matrix, ones_matrix], [zeros_matrix, target_attend_matrix]])
    return mask_matrix


def train(epoch, input_train_data):
    global update_number
    model.train()
    all_data = []
    for one_data in input_train_data:
        data = get_shuffle_data(one_data[0])
        lang = one_data[1]
        total = len(data)
        for i in range(0, total, batch_size):
            sample = data[i:i + batch_size]
            context = [x[0][::-1] for x in sample]
            question_source = [[1] + x[1] for x in sample]
            question_target = [x[1] + [2] for x in sample]
            context, _ = padding(context, max_len=doc_max_length_size)
            question_source, _ = padding(question_source, max_len=question_max_length_size)
            question_target, _ = padding(question_target, max_len=question_max_length_size)
            seq = torch.LongTensor(np.concatenate([np.flip(context, 1), question_source], 1)).cuda()
            target_idx_mask = torch.BoolTensor(
                np.concatenate([np.zeros_like(context), np.ones_like(question_source)], 1)).cuda()
            mask_attention_index = torch.ByteTensor(
                get_partial_attention_matrix(context.shape[1], question_source.shape[1])).cuda()
            question_target = torch.LongTensor(question_target).cuda().flatten()
            all_data.append([seq, mask_attention_index, target_idx_mask, question_target, lang])
    total_loss = 0
    num = 0
    pre_time = None
    instance_number = 0
    total = sum([x[0].size(0) for x in all_data])
    i = 0
    lst = list(range(len(all_data)))
    np.random.shuffle(lst)
    for j in lst:
        [seq, mask_attention_index, target_idx_mask, question_target, lang] = all_data[j]
        i += seq.size(0)
        loss = model([seq, mask_attention_index, target_idx_mask, question_target], lang)
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()
        # torch.nn.utils.clip_grad_norm_(amp.master_params(optim), 0.1)
        total_loss += loss.item() * seq.size(0)
        instance_number += seq.size(0)
        if num % iters_to_accumulate == 0:
            optim.step()
            optim.zero_grad()
            update_number += 1
        num += 1
        if num % log_interval == 0:
            if pre_time is None:
                eclipse = 0
            else:
                eclipse = time.time() - pre_time
            total_loss = metric_sum(total_loss)
            instance_number = metric_sum(instance_number)
            if dist.get_rank() == 0:
                print(
                    'epoch {}, mask loss is {:5.4f}, ms per batch is {:7.5f}, eclipse {:4.3f}%  lr={:e}'.format(epoch,
                                                                                                                total_loss / instance_number,
                                                                                                                1000 * eclipse / instance_number,
                                                                                                                i * 100 / total,
                                                                                                                optim.param_groups[
                                                                                                                    0][
                                                                                                                    'lr']))
        pre_time = time.time()
        total_loss = 0
        instance_number = 0


def upsampling_data(input_data):
    max_size = max([len(x[0]) for x in input_data])
    for x in input_data:
        data_size = len(x[0])
        sample_size = int((max_size - data_size)*0.5)
        extended_data = random.choices(x[0], k=sample_size)
        x[0].extend(extended_data)


if __name__ == '__main__':
    random.seed(2042)
    # one_train_data = [train_data[primary], train_data[secondary]]
    one_train_data = [[load_file(x[0]), x[1]] for x in train_data]
    if dist.get_rank() == 0:
        print('load training data ..., current size is {}'.format(sum([len(x[0]) for x in one_train_data])))
    upsampling_data(one_train_data)
    if dist.get_rank() == 0:
        print(sum([len(x[0]) for x in one_train_data]))
    languages = [x[1] for x in one_train_data]
    for i in range(0, 10):
        train(i, one_train_data)
        if dist.get_rank() == 0:
            output = {}
            for name, param in model.module.named_parameters():
                output[name] = param.data.cpu().numpy()
            dump_file(output, '../model/model.qg.{}.all.{}.th'.format(n_hidden, i))
