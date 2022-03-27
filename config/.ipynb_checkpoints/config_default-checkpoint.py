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
        'vocab_min': 1,
        'padding_word': '<PADDING>',
        'padding_id': 0
    },
    'model': {
        'save_dir_path': '/Users/yejianfeng/Desktop/model_dir'
    },
    'contexts': {
        'max_window_size': 31
    },
    'max_co_change_num': 30,
}


def get_config():
    return configs
