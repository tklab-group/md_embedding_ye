import sys
sys.path.append('../')
from data.mongo import MethodMapDao
from data.mongo import ModuleDataDao
import numpy as np
from config.config_default import get_config
from data.vocabulary import Vocabulary
from common.util import load_data, save_data
from statistics import mean, median, stdev, variance
# def padding_contexts(contexts):
#     max_contents_ws = 0
#     for i in range(len(contexts)):
#         contents_ws = len(contexts[i])
#         if contents_ws > max_contents_ws:
#             max_contents_ws = contents_ws
#     # padding
#     for i in range(len(contexts)):
#         for j in range(max_contents_ws - len(contexts[i])):
#             contexts[i].append(-1)
#     return contexts


# def get_contexts_target(git_name):
#     dao = ExperimentDao()
#     experiment_list = dao.query(git_name)
#     train_list = experiment_list['train']
#     validate_list = experiment_list['validate']
#     train_contexts = []
#     train_target = []
#     validate_contexts = []
#     validate_target = []
#     for i in range(len(train_list)):
#         list = train_list[i]['list']
#         for j in range(len(list)):
#             if len(list[j]['c']) == 0:
#                 continue
#             train_contexts.append(list[j]['c'])
#             train_target.append(list[j]['t'])
#     for i in range(len(validate_list)):
#         list = validate_list[i]['list']
#         for j in range(len(list)):
#             validate_contexts.append(list[j]['c'])
#             validate_target.append(list[j]['t'])
#     train = {'contexts': padding_contexts(train_contexts), 'target': train_target}
#     validate = {'contexts': padding_contexts(validate_contexts), 'target': validate_target}
#     return train, validate

def get_module_data(git_name):
    dao = ModuleDataDao()
    module_data = dao.query(git_name)
    return module_data['list']


def get_method_map(git_name):
    dao = MethodMapDao()
    method_map = dao.query(git_name)
    return method_map['list']


def save_module_method_map_pkl(git_name):
    module_data = get_module_data(git_name)
    method_map_data = get_method_map(git_name)
    data = {
        'module_data': module_data,
        'method_map_data': method_map_data
    }
    pkl_file_path = './model_params/pre_load_' + git_name + '_v1.0.0.pkl'
    save_data(data, pkl_file_path)


def load_module_method_map_pkl(git_name):
    pkl_file_path = './model_params/pre_load_' + git_name + '_v1.0.0.pkl'
    data = load_data(pkl_file_path)
    return data['module_data'], data['method_map_data']


def leave_one_out(transaction):
    pair_list = []
    size = len(transaction)
    idx = np.arange(1, size) - np.tri(size, size - 1, k=-1, dtype=bool)
    query_list = np.array(transaction)[idx]
    length = len(query_list)
    for i in range(length):
        pair_list.append({
            'contexts': query_list[i][:],
            'target': transaction[i]
        })
    return pair_list


class DataLoader:
    # mode normal sub-word kobayashi(sub-word without fullname) n-gram
    def __init__(self, git_name, validate_data_num, is_load_from_pkl=False, mode="normal"):
        config_all = get_config()
        self.config = config_all['contexts']
        self.train_data = {}
        self.validate_data = {}
        # all_corpus all_word_to_id all_id_to_word是对应所有的数据的
        self.all_corpus = []
        self.all_word_to_id = {}
        self.all_id_to_word = {}
        self.git_name = ''
        self.validate_data_num = 0
        self.validate_end_count = 0
        self.module_data = []
        self.padding_id = config_all['dataset']['padding_id']
        self.mode = mode

        # init
        self.git_name = git_name
        self.validate_data_num = validate_data_num
        if is_load_from_pkl:
            print('load from pkl')
            self.module_data, self.method_map_data = load_module_method_map_pkl(git_name)
        else:
            self.module_data = get_module_data(git_name)
            self.method_map_data = get_method_map(git_name)
        self.build_all_word_id_map()
        self.cal_validate_end_count()
        self.vocab = Vocabulary(self.module_data[self.validate_end_count:],
                                self.all_id_to_word, self.all_word_to_id, self.mode)
        self.build_train_data()
        self.build_validate_data()

    def build_all_word_id_map(self):
        for i in range(len(self.method_map_data)):
            word = self.method_map_data[i]['item']
            if self.mode == 'n-gram':
                word = '<' + word + '>'
            word_id = self.method_map_data[i]['index']
            self.all_corpus.append(word)
            self.all_word_to_id[word] = word_id
            self.all_id_to_word[word_id] = word

    def cal_validate_end_count(self):
        # 记录当前用到的数量
        validate_count = 0
        # 记录总共用到的数量
        validate_end_count = 0
        # for validate data[0, validate_data_num-1]
        for i in range(len(self.module_data)):
            validate_end_count += 1
            module_data_item = self.module_data[i]['list']
            if len(module_data_item) <= 1 or len(module_data_item) > self.config['max_window_size']:
                continue
            validate_count += 1
            # 如果拿到目标预定的数量，那就结束遍历
            if validate_count == self.validate_data_num:
                break
        self.validate_end_count = validate_end_count

    def get_extension_by_md_id(self, contexts):
        if self.mode == 'normal':
            return contexts
        word_id_list = []
        for i in range(len(contexts)):
            md_id = contexts[i]
            if md_id in self.vocab.md_id_to_word_id:
                word_id = self.vocab.md_id_to_word_id[md_id]
                word_id_list.append(word_id)
        return self.get_extension_by_word_id(word_id_list)

    def get_extension_by_word_id(self, contexts):
        if self.mode == 'normal':
            return contexts
        result = []
        for i in range(len(contexts)):
            word_id = contexts[i]
            if self.mode == 'n-gram':
                n_grams_id_list = self.vocab.word_id_to_n_grams_id_list[word_id]
                result.extend(n_grams_id_list)
            if self.mode == 'kobayashi' or self.mode == 'sub-word':
                sub_word_id_list = self.vocab.word_id_to_sub_words_id_list[word_id]
                result.extend(sub_word_id_list)
        if self.mode == 'n-gram' or self.mode == 'sub-word':
            result.extend(contexts)
        return result

    def build_train_data(self):
        if self.validate_data_num > len(self.module_data):
            return

        train_contexts = []
        train_target = []

        # 这里顺序要反过来
        # for train data[validate_count len(module_data)-1]
        for i in range(len(self.module_data) - self.validate_end_count - 1, -1, -1):
            real_index = i + self.validate_end_count
            module_data_item = self.module_data[real_index]['list']

            if len(module_data_item) <= 1 or len(module_data_item) > self.config['max_window_size']:
                continue
            result = self.create_leave_one_out_contexts_target(transaction=module_data_item,
                                                               md_id_to_word_id=self.vocab.md_id_to_word_id)
            if len(result) != len(module_data_item):
                print(module_data_item)
                return
            for j in range(len(result)):
                q = result[j]['contexts']
                # 根据mode进行拓展
                q = self.get_extension_by_word_id(q)
                # padding TODO: 这里如果是拓展过的话，设定的max_window_size就需要弄大点，因为拓展出来的数组会很长的样子
                basic_number = 1
                if self.mode != 'normal':
                    basic_number = 10
                if basic_number * self.config['max_window_size'] - len(q) > 0:
                    padding_array = np.arange(self.config['max_window_size'] - len(q), dtype=int)
                    padding_array = np.full_like(padding_array, self.padding_id)
                    q = np.concatenate((q, padding_array), axis=0)
                train_contexts.append(q)
                train_target.append(result[j]['target'])
        # print('train_contexts', train_contexts)
        # print('train_target', train_target)
        # print('validate_contexts', validate_contexts)
        # print('validate_target', validate_target)
        # 这里把顺序给反过来
        train_data = {'contexts': train_contexts, 'target': train_target}
        # print('validate_count', validate_count)
        # print('validate_end_count', validate_end_count)
        self.train_data = train_data
        return train_data

    # 验证数据这里不进行拓展，让Evaluation类那边自己拓展，便于统计数据
    def build_validate_data(self):
        validate_contexts = []
        validate_target = []
        # for validate data[0, validate_data_num-1]
        # 这里的create_leave_one_out_contexts_target(module_data_item, self.vocab.md_id_to_word_id)会报错
        # 原因很简单，因为词库压根没用validate的数据
        # 这里顺序要反过来
        for i in range(self.validate_end_count - 1, -1, -1):
            module_data_item = self.module_data[i]['list']
            # 验证数据也要进行限制，如果过多md变动的话，本身就没有必要验证
            if len(module_data_item) <= 1 or len(module_data_item) > self.config['max_window_size']:
                continue
            result = self.create_leave_one_out_contexts_target(transaction=module_data_item)
            # print('len', len(predict.pair_list), len(module_data_item))
            if len(result) != len(module_data_item):
                print(module_data_item)
                return
            for j in range(len(result)):
                validate_contexts.append(result[j]['contexts'])
                validate_target.append(result[j]['target'])
        validate_data = {'contexts': validate_contexts, 'target': validate_target}
        self.validate_data = validate_data
        return validate_data

    # transaction里面都是md的id
    def create_leave_one_out_contexts_target(self, transaction, md_id_to_word_id={}):
        # print('md_id_to_word_id', md_id_to_word_id)
        pair_list = []
        size = len(transaction)
        # 如果md_id_to_word_id有作为参数传入
        if len(md_id_to_word_id.keys()) > 0:
            # 替换id
            for i in range(size):
                # print('transaction[i]', transaction[i])
                # print('is', transaction[i] in md_id_to_word_id)
                transaction[i] = md_id_to_word_id[transaction[i]]
        return leave_one_out(transaction)

    # 统计训练数据和验证数据里面平均的module data数量
    def count_average_num(self):
        count_list = []
        validate_count_list = []
        train_count_list = []
        # 验证数据
        for i in range(self.validate_end_count - 1, -1, -1):
            module_data_item = self.module_data[i]['list']
            if len(module_data_item) <= 1 or len(module_data_item) > self.config['max_window_size']:
                continue
            len_data = len(module_data_item)
            count_list.append(len_data)
            validate_count_list.append(len_data)
        # 训练数据
        for i in range(len(self.module_data) - self.validate_end_count - 1, -1, -1):
            real_index = i + self.validate_end_count
            module_data_item = self.module_data[real_index]['list']
            if len(module_data_item) <= 1 or len(module_data_item) > self.config['max_window_size']:
                continue
            len_data = len(module_data_item)
            count_list.append(len_data)
            train_count_list.append(len_data)
        # 统计数据
        # print('count_list', count_list)
        # print('validate_count_list', validate_count_list)
        # print('train_count_list', train_count_list)
        print('all mean %f | median %f | stdev %f | variance %f'
              % (mean(count_list), median(count_list), stdev(count_list), variance(count_list)))
        print('validate mean %f | median %f | stdev %f | variance %f'
              % (mean(validate_count_list), median(validate_count_list), stdev(validate_count_list),
                 variance(validate_count_list)))
        print('train mean %f | median %f | stdev %f | variance %f'
              % (mean(train_count_list), median(train_count_list), stdev(train_count_list), variance(train_count_list)))

    def cal_use_commit_count(self):
        # 记录当前用到的数量
        can_use_count = 0
        for i in range(len(self.module_data)):
            module_data_item = self.module_data[i]['list']
            if len(module_data_item) <= 1 or len(module_data_item) > self.config['max_window_size']:
                continue
            can_use_count += 1
        return can_use_count

    # def build_vocab(self):
    #     self.vocab.build(self.module_data[self.validate_end_count:], self.all_id_to_word, self.all_word_to_id)

    # def get_dataset(self):
    #     return self.dataset

    # def get_triplet(self):
    #     dao = TripletDao()
    #     triplet_list = dao.query_all()
    #     md_list = []
    #     p_list = []
    #     n_list = []
    #     for triplet in triplet_list:
    #         md_list.append(triplet['md'])
    #         p_list.append(triplet['p'])
    #         n_list.append(triplet['n'])
    #     # 这里需要把md,p,n都拿出来当做是三个不同数组然后同时放到dataset里面
    #     data_set = tf.data.Dataset.from_tensor_slices((md_list, p_list, n_list))
    #     data_set = data_set.shuffle(self.config['train_shuffle_buffer_size']).batch(
    #         self.config['train_batch_size'])
    #     return data_set
