# -*- coding: utf-8 -*-
"""
 @Time    : 2020/5/21 下午6:37
 @FileName: pre_train_lalm.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import argparse
import sys
import time

sys.path.append('..')
import apex

from models.LALM import LALM
from utils import *
import multiprocessing as mp
import torch.distributed as dist

from apex import amp
import sentencepiece as spm

torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')

# Pin GPU to be used to process local rank (one GPU per process)
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--en_path", type=str, help='english wiki path')
parser.add_argument("--zh_path", type=str, help='chinese wiki path')
parser.add_argument("--ko_path", type=str, help='korean wiki path')
parser.add_argument("--fr_path", type=str, help='french wiki path')
parser.add_argument("--hi_path", type=str, help='hindi wiki path')
parser.add_argument("--bu_path", type=str, help='burmese wiki path')
parser.add_argument("--de_path", type=str, help='german wiki path')
parser.add_argument("--vi_path", type=str, help='vietnam wiki path')
parser.add_argument("--ja_path", type=str, help='japanese wiki path')
parser.add_argument("--mi_path", type=str, help='chinese minnan wiki path')
parser.add_argument("--reload", help='reload pretrained parameters', action="store_true", default=False)
parser.add_argument("--epoch", type=int, help='reload pretrained parameters epoch', default=0)
parser.add_argument("--batch_size", type=int, help='batch size', default=8)
parser.add_argument("--max_learning_rate", type=float, help='maximum learning rate', default=1e-4)
parser.add_argument("--max_length", type=int, help='max text length to be processed', default=512)
parser.add_argument("--n_embedding", type=int, help='embedding size', default=128)
parser.add_argument("--n_hidden", type=int, help='hidden size', default=1024)
parser.add_argument("--n_layer", type=int, help='number of layers', default=24)
parser.add_argument("--n_head", type=int, help='number of heads', default=16)
parser.add_argument("--type", type=str, help='types of the LALM model, could be shared or lalm', default='shared')
args = parser.parse_args()
print(args.reload)
torch.cuda.set_device(args.local_rank)

n_embedding = args.n_embedding
n_hidden = args.n_hidden
n_layer = args.n_layer
n_head = args.n_head
batch_size = args.batch_size
max_learning_rate = args.max_learning_rate

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

zh_file_paths = [args.zh_path]
en_file_paths = [args.en_path]
ko_file_paths = [args.ko_path]
fr_file_paths = [args.fr_path]
hi_file_paths = [args.hi_path]
bu_file_paths = [args.bu_path]
de_file_paths = [args.de_path]
vi_file_paths = [args.vi_path]
ja_file_paths = [args.ja_path]
mi_file_paths = [args.mi_path]

file_paths = [[x, 0] for x in en_file_paths] + \
             [[x, 1] for x in zh_file_paths] + \
             [[x, 2] for x in ko_file_paths] + \
             [[x, 3] for x in fr_file_paths] + \
             [[x, 4] for x in hi_file_paths] + \
             [[x, 5] for x in bu_file_paths] + \
             [[x, 6] for x in de_file_paths] + \
             [[x, 7] for x in vi_file_paths] + \
             [[x, 8] for x in ja_file_paths] + \
             [[x, 9] for x in mi_file_paths]

log_interval = 256
# iters_to_accumulate = 4096 // (batch_size * dist.get_world_size())
iters_to_accumulate = 1
warm_up_steps = 5000
decay_steps = 300000
d_a, d_b, d_c = 0.8, 0.1, 0.1
num_thread = len(file_paths)
max_length_size = args.max_length + 1

# print('model size {}'.format(get_model_parameters(model)))
queue = mp.Queue(10)


def get_line_id(line, sp):
    ids = line.strip()
    return sp.encode_as_ids(ids) + [2]


def get_partial_attention_matrix(source_sentence_length, target_sentence_length):
    source_attend_matrix = np.zeros([source_sentence_length, source_sentence_length])
    target_attend_matrix = np.triu(np.ones([target_sentence_length, target_sentence_length]), 1)
    ones_matrix = np.ones([source_sentence_length, target_sentence_length])
    zeros_matrix = np.zeros([target_sentence_length, source_sentence_length])
    mask_matrix = np.block([[source_attend_matrix, ones_matrix], [zeros_matrix, target_attend_matrix]])
    return mask_matrix


def deal_one_batch(batch):
    seq_len = batch.shape[1]
    b_size = batch.shape[0]
    inputs_type = random.sample([0, 1, 2], 1)[0]
    # inputs_type = 1
    if inputs_type == 0:  # uni-gram language model
        return [torch.LongTensor(batch)]
    elif inputs_type == 1:  # mask LM
        seq_len -= 1
        size = int(seq_len * 0.15)
        output = []
        indexes = []
        for i in range(b_size):
            sample_prob = None
            index = np.sort(np.random.choice(seq_len, size=size, replace=False, p=sample_prob))
            indexes.append(index)
            tmp = []
            for j in index:
                prob = np.random.rand()
                if prob <= d_a:
                    tmp.append(batch[i][j])
                    batch[i][j] = 30002
                elif prob <= d_a + d_b:
                    tmp.append(batch[i][j])
                    batch[i][j] = np.random.randint(0, 30003)
                else:
                    tmp.append(batch[i][j])
            output.append(tmp)

        return [torch.LongTensor(batch)[:, 0:-1], torch.LongTensor(indexes), torch.LongTensor(output)]
    else:  # seq to seq
        source_sentence_length = np.random.random_integers(int(seq_len / 1.5), seq_len - 5)
        target_sentence_length = seq_len - source_sentence_length - 1
        batch = np.insert(batch, source_sentence_length, 1, axis=1)
        attention_mask = get_partial_attention_matrix(source_sentence_length, target_sentence_length)
        source = batch[:, 0:-2]
        target = batch[:, -target_sentence_length - 1:-1]
        target_index = torch.BoolTensor([[0] * source_sentence_length + [1] * target_sentence_length] * batch_size)
        return [torch.LongTensor(source), torch.ByteTensor(attention_mask), target_index,
                torch.LongTensor(target).flatten()]


def generate_data(thread_id):
    np.random.seed((dist.get_rank() * num_thread + thread_id) * 34 + 643)
    np.random.shuffle(file_paths)
    while True:
        for file_path_id in file_paths:
            file_path = file_path_id[0]
            lang = file_path_id[1]
            if lang != thread_id:
                continue
            data = []
            num = -1
            charset = get_file_charset(file_path)
            with open(file_path, encoding=charset, errors='ignore') as f:
                for line in f:
                    num += 1
                    if num % num_thread != dist.get_rank():
                        continue
                    if np.random.binomial(1, 0.1):
                        continue
                    cc = get_line_id(line, sp_model[lang])
                    data.extend(cc)
                    if len(data) >= batch_size * max_length_size:
                        seq = np.asarray(data[:batch_size * max_length_size]).reshape(batch_size, max_length_size)
                        mask_data = deal_one_batch(seq)
                        queue.put(mask_data + [lang])
                        data = data[batch_size * max_length_size - len(data):]
            if dist.get_rank() == 0:
                print('rank {} is done for thread {} for {}'.format(dist.get_rank(), thread_id, file_path))


for one in range(num_thread):
    p = mp.Process(target=generate_data, args=(one,))
    p.start()

model = LALM(vocab_size_lst, n_embedding, n_hidden, n_layer, n_head)
epoch = args.epoch
reload = args.reload
if reload:
    state_dict = load_file('../model/model.xunilm.{}.{}.{}.th'.format(n_hidden, n_layer, epoch))
    for name, para in model.named_parameters():
        if name not in state_dict:
            print('{} not load'.format(name))
            continue
        para.data = torch.FloatTensor(state_dict[name])
print('model size {}'.format(get_model_parameters(model)))
model = model.cuda()
optimizer = apex.optimizers.FusedAdam(model.parameters(),
                                      weight_decay=0.01,
                                      lr=1.0e-7)

model, optimizer = amp.initialize(model,
                                  optimizer,
                                  opt_level="O2",
                                  loss_scale='dynamic',
                                  keep_batchnorm_fp32=False,
                                  verbosity=0
                                  )

model = apex.parallel.DistributedDataParallel(model)
# model = torch.nn.parallel.DistributedDataParallel(model)
lr_opt_steps = max_learning_rate / decay_steps
warm_up_lr_opt_steps = max_learning_rate / warm_up_steps

test_data = []
num = 0
print('test data size is {}'.format(len(test_data)))


def metric_sum(val):
    tensor = torch.tensor(val).cuda()
    dist.reduce(tensor, 0)
    return tensor.item()


def metric_average(val):
    tensor = torch.tensor(val).cuda()
    dist.reduce(tensor, 0)
    return tensor.item() / dist.get_world_size()


def test():
    model.eval()
    result = []
    with torch.no_grad():
        for j in range(0, len(test_data), batch_size * max_length_size):
            # while not check_program_status():
            #     time.sleep(1)
            seq = np.asarray(test_data[j:j + batch_size * max_length_size]).reshape(batch_size, max_length_size)
            mask_data = deal_one_batch(seq)
            _, output = model(mask_data)
            result.append(output.item())
    return metric_average(np.mean(result))


update_number = 0
current_number = 0


def train(sent_processed=0):
    global update_number, current_number
    model.train()
    num = 0
    total_loss = 0
    pre_time = None
    current_rank_processed = 0
    while True:
        one = queue.get()
        language = one[-1]
        sample = one[:-1]
        loss = model([x.cuda() for x in sample], language)
        for a_param in model.module.embedding.parameters():
            loss += 1.0e-8 * torch.norm(a_param)
        loss /= iters_to_accumulate
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if update_number % iters_to_accumulate == 0:
            # Every iters_to_accumulate iterations, unscale and step
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 0.1)
            optimizer.step()
            optimizer.zero_grad()
            for param_group in optimizer.param_groups:
                if current_number > warm_up_steps:
                    param_group['lr'] -= lr_opt_steps
                else:
                    param_group['lr'] += warm_up_lr_opt_steps
            current_number += 1
            if current_number > (warm_up_steps + decay_steps):
                current_number = 0
        current_rank_processed += one[0].size(0)
        total_loss += loss.item() * one[0].size(0) * iters_to_accumulate
        num += 1
        update_number += 1
        if update_number % log_interval == 0:
            # torch.cuda.empty_cache()
            # for param_group in optimizer.param_groups and sent_processed < log_interval * 5000:
            #     if param_group['lr'] < max_learning_rate:
            #         param_group['lr'] = param_group['lr'] * lr_scale
            current_rank_processed = metric_sum(current_rank_processed)
            sent_processed += current_rank_processed
            total_loss_mask_ = metric_sum(total_loss)

            if pre_time is None:
                eclipse = 0
            else:
                eclipse = time.time() - pre_time
            if dist.get_rank() == 0:
                print(
                    '{} mask loss is {:5.4f} ms per sentence is {:7.4f},lr={:e} sent processed {:g}'.format(
                        time.strftime("%Y-%m-%d-%H:%M:%S"),
                        total_loss_mask_ / current_rank_processed,
                        1000 * eclipse / current_rank_processed,
                        optimizer.param_groups[0]['lr'],
                        sent_processed))
            pre_time = time.time()
            total_loss = 0.0
            current_rank_processed = 0
        if num == log_interval * 20:
            break
    return sent_processed


best_acc = 10000000
n = 0
patience = 0
d = 0
for epoch in range(100000):
    n = train(n)
    ppl = 10000
    if dist.get_rank() == 0:
        output = {}
        for name, param in model.module.named_parameters():
            output[name] = param.data.cpu().numpy()
        dump_file(output, '../model/model.xunilm.{}.{}.{}.th'.format(n_hidden, n_layer, epoch // 10))
        print('-------------------------epoch {} ---------------------'.format(epoch))
