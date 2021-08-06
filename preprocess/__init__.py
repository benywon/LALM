from utils import *

batch_size = 16
train_data = [
    (load_file('../data/train.en.obj'), 0),
    (load_file('../data/train.zh.obj'), 1),
    (load_file('../data/train.fr.obj'), 3)
]


def get_shuffle_data(data):
    dist_data = []
    for one_data in data:
        lang = one_data[1]
        whole_data = sorted(one_data[0], key=lambda x: len(x[0] + x[1]))
        remove_data_size = len(whole_data) % 8
        thread_data = [whole_data[x + 4] for x in
                       range(0, len(whole_data) - remove_data_size, 8)]
        total = len(thread_data)
        for i in range(0, total, batch_size):
            sample = data[i:i + batch_size]
            context = [x[0][::-1] for x in sample]
            question_source = [[1] + x[1] for x in sample]
            question_target = [x[1] + [2] for x in sample]
            context, _ = padding(context, max_len=512)
            question_source, _ = padding(question_source, max_len=20)
            question_target, _ = padding(question_target, max_len=20)
            seq = torch.LongTensor(np.concatenate([np.flip(context, 1), question_source], 1))
            target_idx_mask = torch.BoolTensor(
                np.concatenate([np.zeros_like(context), np.ones_like(question_source)], 1))
            mask_attention_index = torch.FloatTensor(
                get_partial_attention_matrix(context.shape[1], question_source.shape[1]))
            question_target = torch.LongTensor(question_target).flatten()
            dist_data.append([])
    return dist_data


get_shuffle_data(train_data)
