#!bin/bash

python3 -m torch.distributed.launch --nproc_per_node=4 pre_train_lalm.py \
--epoch=-1 \
--zh_path='/search/odin/bingning/data/LALM/chinese/wiki.all.txt' \
--en_path='/search/odin/bingning/data/LALM/english/wiki.all.txt' \
--ko_path='/search/odin/bingning/data/LALM/korean/wiki.all.txt' \
--fr_path='/search/odin/bingning/data/LALM/french/wiki.all.txt' \
--hi_path='/search/odin/bingning/data/LALM/hindi/wiki.all.txt' \
--bu_path='/search/odin/bingning/data/LALM/burmese/wiki.all.txt' \
--de_path='/search/odin/bingning/data/LALM/german/wiki.all.txt' \
--vi_path='/search/odin/bingning/data/LALM/vietnam/wiki.all.txt' \
--ja_path='/search/odin/bingning/data/LALM/japanese/wiki.all.txt' \
--mi_path='/search/odin/bingning/data/LALM/minnan/wiki.all.txt'