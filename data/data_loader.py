import sys

sys.path.append('../')
import numpy as np
from config.config_default import get_config
from data.vocabulary import Vocabulary
from statistics import mean, median, stdev, variance
from data.mode_enum import Mode
from data.contexts_target_builder import ContextsTargetBuilder
from data.data_divider import DataDivider
from data.id_mapped import IdMapped
from data.util import get_module_data, get_method_map, save_module_method_map_pkl, load_module_method_map_pkl
from data.freq_counter import FreqCounter
from data.embedding_index_mapped import EmbeddingIndexMapped
from data.sub_sampling import SubSampling
from data.negative_sampling import NegativeSampling
from data.max_metric import MaxMetric
import time
from data.data_store import DataStore
from data.pre_process import PreProcess
from data.rename_chain import RenameChain
from data.delete_record import DeleteRecord


class DataLoader:
    def __init__(self,
                 git_name,
                 expected_validate_length,
                 most_recent,
                 dataStore: DataStore,
                 deleteRecord: DeleteRecord,
                 is_load_from_pkl=False,
                 mode=Mode.NORMAL,
                 is_test=False,
                 test_md_list=None,
                 test_method_map=None,
                 is_sub_sampling=False,
                 is_subword_sub_sampling=False,
                 is_negative_sampling=False,
                 is_check_rename=True,
                 is_contexts_extend=False,
                 is_use_package=True,
                 is_use_class_name=True,
                 is_use_return_type=True,
                 is_use_method_name=True,
                 is_use_param_type=True,
                 is_use_param_name=True,
                 is_split_train_data=False,
                 is_simple_handle_package=False,
                 is_simple_handle_class_name=False,
                 is_simple_handle_return_type=False,
                 is_simple_handle_method_name=False,
                 is_simple_handle_param_type=False,
                 is_simple_handle_param_name=False,
                 is_predict_with_file_level=False,
                 is_mark_respective_type=False,
                 # check preprocessing
                 is_preprocessing_package=True,
                 is_delete_modifier=True,
                 is_delete_void_return_type=True,
                 is_casing=True,
                 is_delete_single_subword=True,
                 is_delete_number_from_method_and_param=False,
                 is_number_type_token_from_return_and_param_type=False,
                 is_delete_sub_word_number=True,
                 ):
        # record time
        start_time = time.time()

        self.git_name = git_name
        self.expected_validate_length = expected_validate_length
        self.most_recent = most_recent
        self.deleteRecord = deleteRecord
        self.is_load_from_pkl = is_load_from_pkl
        self.mode = mode
        self.is_sub_sampling = is_sub_sampling
        self.is_subword_sub_sampling = is_subword_sub_sampling
        self.is_negative_sampling = is_negative_sampling
        self.is_check_rename = is_check_rename
        self.is_contexts_extend = is_contexts_extend
        self.is_use_package = is_use_package
        self.is_use_class_name = is_use_class_name
        self.is_use_return_type = is_use_return_type
        self.is_use_method_name = is_use_method_name
        self.is_use_param_type = is_use_param_type
        self.is_use_param_name = is_use_param_name
        self.is_split_train_data = is_split_train_data
        self.is_simple_handle_package = is_simple_handle_package
        self.is_simple_handle_class_name = is_simple_handle_class_name
        self.is_simple_handle_return_type = is_simple_handle_return_type
        self.is_simple_handle_method_name = is_simple_handle_method_name
        self.is_simple_handle_param_type = is_simple_handle_param_type
        self.is_simple_handle_param_name = is_simple_handle_param_name
        self.is_predict_with_file_level = is_predict_with_file_level
        self.is_mark_respective_type = is_mark_respective_type
        self.is_preprocessing_package = is_preprocessing_package
        self.is_delete_modifier = is_delete_modifier
        self.is_delete_void_return_type = is_delete_void_return_type
        self.is_casing = is_casing
        self.is_delete_single_subword = is_delete_single_subword
        self.is_delete_number_from_method_and_param = is_delete_number_from_method_and_param
        self.is_number_type_token_from_return_and_param_type = is_number_type_token_from_return_and_param_type
        self.is_delete_sub_word_number = is_delete_sub_word_number

        # config
        config_all = get_config()
        self.config_all = config_all
        self.padding_id = config_all['dataset']['padding_id']
        self.vocab_min = config_all['dataset']['vocab_min']

        self.load_data_start_time = time.time()
        # load data from database
        if not is_test:
            self.md_list = dataStore.get_module_data(git_name)
            self.method_map = dataStore.get_method_map(git_name)
            self.rename_chain_data = dataStore.get_rename_chain(git_name)
        else:
            self.md_list = test_md_list
            self.method_map = test_method_map
            self.rename_chain_data = []
        self.delete_record_list = dataStore.get_delete_record(git_name)
        self.load_data_cost_time = time.time() - self.load_data_start_time

        self.data_diver_start_time = time.time()
        # divide data into train data and validate data
        self.dataDivider = DataDivider(self.md_list, expected_validate_length, most_recent)
        self.data_diver_cost_time = time.time() - self.data_diver_start_time

        # test
        self.maxMetric_false = MaxMetric(self.dataDivider, False)
        self.maxMetric_true = MaxMetric(self.dataDivider, True)

        self.id_mapped_start_time = time.time()
        # manager id map
        self.idMapped = IdMapped(self.dataDivider.get_train_data(), self.method_map)
        self.id_mapped_cost_time = time.time() - self.id_mapped_start_time

        self.renameChain = RenameChain(
            self.dataDivider.get_validate_data(),
            self.dataDivider.validate_data_commit_hash_list,
            self.idMapped.all_id_to_word,
            self.idMapped.all_word_to_id,
            self.rename_chain_data
        )

        self.pre_process_start_time = time.time()
        self.preProcess = PreProcess(
            train_data=self.dataDivider.get_train_data(),
            train_id_to_word=self.idMapped.train_id_to_word,
            train_words=self.idMapped.train_words,
            train_data_commit_hash_list=self.dataDivider.train_data_commit_hash_list,
            renameChain=self.renameChain,
            is_check_rename=self.is_check_rename,
            mode=mode,
            is_use_package=self.is_use_package,
            is_use_class_name=self.is_use_class_name,
            is_use_return_type=self.is_use_return_type,
            is_use_method_name=self.is_use_method_name,
            is_use_param_type=self.is_use_param_type,
            is_use_param_name=self.is_use_param_name,
            is_simple_handle_package=self.is_simple_handle_package,
            is_simple_handle_class_name=self.is_simple_handle_class_name,
            is_simple_handle_return_type=self.is_simple_handle_return_type,
            is_simple_handle_method_name=self.is_simple_handle_method_name,
            is_simple_handle_param_type=self.is_simple_handle_param_type,
            is_simple_handle_param_name=self.is_simple_handle_param_name,
            is_predict_with_file_level=self.is_predict_with_file_level,
            is_mark_respective_type=self.is_mark_respective_type,
            is_preprocessing_package=self.is_preprocessing_package,
            is_delete_modifier=self.is_delete_modifier,
            is_delete_void_return_type=self.is_delete_void_return_type,
            is_casing=self.is_casing,
            is_delete_single_subword=self.is_delete_single_subword,
            is_delete_number_from_method_and_param=self.is_delete_number_from_method_and_param,
            is_number_type_token_from_return_and_param_type=self.is_number_type_token_from_return_and_param_type,
            is_delete_sub_word_number=self.is_delete_sub_word_number
        )
        self.common_prefix = self.preProcess.common_prefix
        self.pre_process_cost_time = time.time() - self.pre_process_start_time

        self.freq_counter_start_time = time.time()
        # counter
        self.freqCounter = FreqCounter(
            train_data=self.dataDivider.get_train_data(),
            train_id_to_word=self.idMapped.train_id_to_word,
            preProcess=self.preProcess,
            mode=mode,
            is_predict_with_file_level=self.is_predict_with_file_level)
        self.freq_counter_cost_time = time.time() - self.freq_counter_start_time

        self.sub_sampling_start_time = time.time()
        # sub sampling
        self.subSampling = None
        if is_sub_sampling or is_subword_sub_sampling:
            self.subSampling = SubSampling(self.freqCounter)
        self.sub_sampling_cost_time = time.time() - self.sub_sampling_start_time

        self.vocab_start_time = time.time()
        # vocabulary
        self.vocab = Vocabulary(self.freqCounter)
        self.vocab_cost_time = time.time() - self.vocab_start_time

        self.embedding_index_mapped_start_time = time.time()
        # manager embedding index map
        self.embeddingIndexMapped = EmbeddingIndexMapped(
            vocab=self.vocab,
            preProcess=self.preProcess,
            mode=mode,
            is_predict_with_file_level=self.is_predict_with_file_level)
        self.embedding_index_mapped_cost_time = time.time() - self.embedding_index_mapped_start_time

        self.negative_sampling_start_time = time.time()
        # negative sampling
        self.negativeSampling = None
        if is_negative_sampling:
            self.negativeSampling = NegativeSampling(
                freqCounter=self.freqCounter,
                idMapped=self.idMapped,
                train_data=self.dataDivider.get_train_data()
            )
            self.negativeSampling.debug_info()
        self.negative_sampling_cost_time = time.time() - self.negative_sampling_start_time

        self.contexts_target_start_time = time.time()
        # context target builder
        self.contextsTargetBuilder = ContextsTargetBuilder(
            embeddingIndexMapped=self.embeddingIndexMapped,
            idMapped=self.idMapped,
            mode=self.mode,
            subSampling=self.subSampling,
            renameChain=self.renameChain,
            is_sub_sampling=self.is_sub_sampling,
            is_subword_sub_sampling=self.is_subword_sub_sampling,
            is_negative_sampling=is_negative_sampling,
            negativeSampling=self.negativeSampling,
            is_check_rename=self.is_check_rename,
            is_contexts_extend=self.is_contexts_extend,
            is_split_train_data=self.is_split_train_data,
            is_predict_with_file_level=self.is_predict_with_file_level
        )
        contexts, target, negative_sampling = self.contextsTargetBuilder.get_train_contexts_target(
            self.dataDivider.get_train_data(),
            self.dataDivider.train_data_commit_hash_list)
        # print('contexts', contexts)
        self.train_contexts_target = {
            'contexts': contexts,
            'target': target,
            'negative_sampling': negative_sampling
        }
        self.contexts_target_cost_time = time.time() - self.contexts_target_start_time

        self.validate_data = self.dataDivider.get_validate_data()

        self.cost_time = time.time() - start_time

    def debug_info(self):
        print('mode, cost time => ', self.mode, self.cost_time)
        print('common_prefix', self.common_prefix)
        print('is_predict_with_file_level', self.is_predict_with_file_level)
        print(self.config_all)
        # self.deleteRecord.debug()
        # print('load_data_cost_time', self.load_data_cost_time)
        # print('data_diver_cost_time', self.data_diver_cost_time)
        # print('id_mapped_cost_time', self.id_mapped_cost_time)
        # print('freq_counter_cost_time', self.freq_counter_cost_time)
        # print('sub_sampling_cost_time', self.sub_sampling_cost_time)
        # print('vocab_cost_time', self.vocab_cost_time)
        # print('embedding_index_mapped_cost_time', self.embedding_index_mapped_cost_time)
        # print('negative_sampling_cost_time', self.negative_sampling_cost_time)
        print('contexts_target_cost_time', self.contexts_target_cost_time)
        print('train and validate data size => ', len(self.dataDivider.get_train_data()),
              len(self.dataDivider.get_validate_data()))
        # mean, median, stdev, variance
        self.dataDivider.count_average_num()
        self.dataDivider.count_oov_rate(True)
        self.dataDivider.count_oov_rate(False)
        self.dataDivider.to_count_record()
        self.dataDivider.check_can_use_commit_size()
        # self.dataDivider.show_count_record_plot()
        print('total using word size => ', self.dataDivider.get_using_module_data_num())
        print('train word size, sub word size, file size=> ',
              len(self.vocab.word_to_index), len(self.vocab.sub_word_to_index), len(self.vocab.file_to_index))
        self.debug_context_info()
        # self.debug_validate_commit_size()
        self.debug_sub_word_vocab()
        # self.debug_sub_word_freq_info()
        # self.maxMetric_false.eval()
        # self.maxMetric_false.eval(False)
        # self.maxMetric_true.eval()
        # self.maxMetric_true.eval(False)
        if self.is_predict_with_file_level:
            file_count_list = []
            for key in self.freqCounter.file_counter:
                # print(key, self.freqCounter.file_counter)
                file_count_list.append(self.freqCounter.file_counter[key])
            print('file pre method count mean %f | median %f | stdev %f | variance %f'
                  % (mean(file_count_list), median(file_count_list), stdev(file_count_list), variance(file_count_list)))

    def debug_context_info(self):
        count_list = []
        print('train data size', len(self.train_contexts_target['contexts']))
        for i in range(len(self.train_contexts_target['contexts'])):
            count = 0
            cur = self.train_contexts_target['contexts'][i]
            for j in range(len(cur)):
                if cur[j] != 0:
                    count += 1
            count_list.append(count)
        print('contexts count mean %f | median %f | stdev %f | variance %f'
              % (mean(count_list), median(count_list), stdev(count_list), variance(count_list)))

    def debug_validate_commit_size(self):
        count_list = []
        print('validate data size', len(self.validate_data))
        for i in range(len(self.validate_data)):
            count_item = len(self.validate_data[i])
            print('count_item', i, count_item)
            count_list.append(count_item)

    def debug_sub_word_freq_info(self):
        if self.mode != Mode.NORMAL:
            counter = self.freqCounter.sub_word_counter
            total_count = self.freqCounter.sub_word_total_count
            print('total_count', total_count)
            freq_list = {word: count / total_count for word, count in counter.items()}
            for word in freq_list:
                print(word, freq_list[word])

    def debug_sub_word_vocab(self):
        if self.mode != Mode.NORMAL:
            sub_words = self.vocab.sub_words
            counter = self.freqCounter.sub_word_counter
            for i in range(len(sub_words)):
                word = sub_words[i]
                if word != '<PADDING>':
                    # print(sub_words[i], counter[sub_words[i]])
                    if word.find('.') > -1 or word.find('-') > -1 or word.find('[') > -1 or word.find(']') > -1 or \
                            word.find('_') > -1 or word.find('#') > -1 or word.find('(') > -1 or word.find(')') > -1:
                        print('weird word', sub_words[i], counter[sub_words[i]])
