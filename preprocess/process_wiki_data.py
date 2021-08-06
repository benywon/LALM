# -*- coding: utf-8 -*-
"""
 @Time    : 2021/6/10 下午12:24
 @FileName: process_wiki_data.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
from utils import *


def check_valid_wiki_passage(txt):
    if txt.startswith('<doc id'):
        return False
    if txt.startswith('</doc>'):
        return False
    if len(txt) < 10:
        return False
    return True


def get_all_wiki_data(language='english'):
    raw_file_paths = get_dir_files('/search/odin/bingning/data/LALM/{}/raw'.format(language))
    data = []
    for one_file in tqdm(raw_file_paths):
        for line in get_file_info(one_file):
            if check_valid_wiki_passage(line):
                data.append(line.strip())
    output_filename = '/search/odin/bingning/data/LALM/{}/wiki.all.txt'.format(language)
    print('dump {} wiki total size is {}'.format(language, len(data)))
    write_lst_to_file(data, output_filename)
    print('{} done!'.format(language))


def train_vocab(language='english', vocab_size=30000):
    sp_path = '/search/odin/bingning/data/LALM/{}/wiki.all.txt'.format(language)
    content = '--input=' + sp_path + ' ' \
                                     '--model_prefix=/search/odin/bingning/data/LALM/language_slot/vocab.my_size --vocab_size=my_size ' \
                                     '--character_coverage=0.9999 ' \
                                     '--num_sub_iterations=2 ' \
                                     '--max_sentencepiece_length=36 ' \
                                     '--model_type=unigram --num_threads=40 --max_sentence_length=15000 ' \
                                     '--input_sentence_size=2000000 '

    content = content.replace('my_size', str(vocab_size))
    content = content.replace('language_slot', language)
    spm.SentencePieceTrainer.Train(
        content
    )


get_all_wiki_data('chinese')
train_vocab('minnan')
