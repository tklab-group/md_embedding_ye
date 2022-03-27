configs = {
    'database': {
        # 'host': '127.0.0.1',
        # for zonzero
        'host': '192.168.202.90',
        'port': 27017,
        'user': 'yejianfeng',
        'password': 'templatedata-ye',
        'database': 'admin'
    },
    'dataset': {
        'train_batch_size': 100,
        'train_shuffle_buffer_size': 10000,
        # ここのvocab_minの値を変更しないしてください、DeleteRecordの実装はvocab_min=1と想定したので
        'vocab_min': 1,
        'sub_vocab_min': 3,
        'file_vocab_min': 1,
        'padding_word': '<PADDING>',
        'padding_id': 0
    },
    'model': {
        'save_dir_path': '/Users/yejianfeng/Desktop/model_dir'
    },
    'max_co_change_num': 29,
    # 'torch_seed': 6,
    'negative_sampling_num': 5,
    'is_load_data_from_pkl': True
}


def get_config():
    return configs
