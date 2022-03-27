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
from data.data_loader import DataLoader
from data.mongo import LowFreqContextsResultDao
from data.low_freq_trace import LowFreqTrace
from data.git_name_version import get_git_name_version
from common.util import save_data, load_data
import os
import random
import torch
from model.main import Main
from model.multi_main import MultiMain


def get_rank(target, top100):
    result = 0
    for i in range(len(top100)):
        if target == top100[i]:
            result = i + 1
            break
    return result


class OOSVDetect:
    def __init__(self):
        # cur_expected_validate_length -> dataLoader
        self.dataLoaderCache = {}

    def detect_for_low_freq_trace(self, dataStore, low_freq_trace_item):
        # print('low_freq_trace_item', low_freq_trace_item)
        git_name = low_freq_trace_item['git_name']
        cur_expected_validate_length = low_freq_trace_item['cur_expected_validate_length']
        most_recent = low_freq_trace_item['most_recent']
        commit_hash = low_freq_trace_item['commit_hash']
        target_word = low_freq_trace_item['predict_result']['target_word']
        target = low_freq_trace_item['predict_result']['target']
        top100_aq = low_freq_trace_item['predict_result']['top100_aq']

        deleteRecord = DeleteRecord(git_name, cur_expected_validate_length)

        # param get model
        is_can_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if is_can_cuda else "cpu")
        main_ent = Main(
            git_name=git_name,
            dim=100,
            batch_size=512,
            max_epoch=1,
            expected_validate_length=cur_expected_validate_length,
            mode=Mode.SUB_WORD,
            dataStore=dataStore,
            deleteRecord=deleteRecord
        )
        train_start_time = time.time()
        main_ent.train(device=device)
        print('train time:', time.time() - train_start_time)

        # get contexts_sub_word_set
        new_list = low_freq_trace_item['new_list']
        old_list = low_freq_trace_item['old_list']
        low_freq_list = low_freq_trace_item['low_freq_list']
        # new_list + old_list + low_freq_list
        contexts = []
        for i in range(len(new_list)):
            contexts.append(new_list[i])
        for i in range(len(old_list)):
            contexts.append(old_list[i])
        for i in range(len(low_freq_list)):
            contexts.append(low_freq_list[i])
        if cur_expected_validate_length in self.dataLoaderCache:
            dataLoader = self.dataLoaderCache[cur_expected_validate_length]
        else:
            dataLoader = main_ent.data_loader
            # cache
            self.dataLoaderCache[cur_expected_validate_length] = dataLoader
        vocab = dataLoader.vocab
        sub_word_to_index = vocab.sub_word_to_index
        # train_id_to_word = dataLoader.idMapped.train_id_to_word
        preProcess = dataLoader.preProcess
        renameChain = dataLoader.renameChain
        contexts_sub_word_list = []
        for i in range(len(contexts)):
            item = contexts[i]
            cur_name = renameChain.get_cur_name_by_hash(commit_hash, int(item))
            sub_word_list = preProcess.get_module_data_sub_word(cur_name)
            # contexts_sub_word_list += sub_word_list
            for j in range(len(sub_word_list)):
                contexts_sub_word_list.append(sub_word_list[j])
        contexts_sub_word_set = set(contexts_sub_word_list)
        analyse_result = self.analyse(target_word, target, contexts_sub_word_set, main_ent)
        self.print_analyse_result(analyse_result)
        return analyse_result

    def print_analyse_result(self, analyse_result):
        print('start print', '-' * 16)
        target_word = analyse_result['target_word']
        print('target_word', target_word)
        for sub_word in analyse_result['contexts']:
            item = analyse_result['contexts'][sub_word]
            score = item['score']
            relative_sub_word_list = item['relative_sub_word_list']
            hit_target_word_list = item['hit_target_word_list']
            print(sub_word)
            print('   score', score)
            print('   relative_sub_word_list')
            for i in range(len(relative_sub_word_list)):
                print('      ', relative_sub_word_list[i])
            print('   hit_target_word_list')
            for i in range(len(hit_target_word_list)):
                print('      ', hit_target_word_list[i])
        print('end print', '-' * 16)

    def analyse(self, target_word, target_index, contexts_sub_word_set, main_ent):
        result = {'target_word': target_word, 'contexts': {}}
        # print(contexts_sub_word_set, target_word)
        dataLoader = main_ent.data_loader
        embeddingIndexMapped = dataLoader.embeddingIndexMapped
        # if target_index in embeddingIndexMapped.out_embedding_index_to_word:
        #     print(embeddingIndexMapped.out_embedding_index_to_word[target_index])
        model = main_ent.model
        sub_word_to_index_map = {}
        index_to_sub_word_map = {}
        for sub_word in contexts_sub_word_set:
            if sub_word in embeddingIndexMapped.word_to_in_embedding_index:
                index = embeddingIndexMapped.word_to_in_embedding_index[sub_word]
                sub_word_to_index_map[sub_word] = index
                index_to_sub_word_map[index] = sub_word
        # print('sub_word index check', len(sub_word_index_list), len(contexts_sub_word_set))
        # score between sub_word and target_word
        index_to_score_map = {}
        # sort
        index_score_triple_list = []
        for index in index_to_sub_word_map:
            score = self.get_score(index, target_index, model)
            index_to_score_map[index] = score
            index_score_triple_list.append(
                (score, index)
            )
        index_score_triple_list.sort(reverse=True)
        # print('index_score_triple_list', index_score_triple_list)
        for i in range(len(index_score_triple_list)):
            triple = index_score_triple_list[i]
            sub_word = index_to_sub_word_map[triple[1]]
            # print('index_score_triple_item', triple[0], triple[1], sub_word)
            result['contexts'][sub_word] = {
                'score': triple[0],
                'relative_sub_word_list': [],
                'hit_target_word_list': []
            }
        for index in index_to_score_map:
            sub_word = index_to_sub_word_map[index]
            score = index_to_score_map[index]
            # print(target_word, sub_word, score)
        # print()
        feature_vector_list, alternate_list = self.build_feature_vector(model, embeddingIndexMapped)
        # print('feature_vector_list', feature_vector_list)
        # print('alternate_list', alternate_list)
        for sub_word_index in index_to_sub_word_map:
            sub_word = index_to_sub_word_map[sub_word_index]
            relative_sub_word_list = \
                self.get_most_relative_sub_word(sub_word_index, sub_word, model, feature_vector_list, alternate_list)
            hit_word_list = self.get_hit_word_list(target_word, relative_sub_word_list)
            result['contexts'][sub_word]['relative_sub_word_list'] = relative_sub_word_list
            result['contexts'][sub_word]['hit_target_word_list'] = hit_word_list
            # print(target_word, sub_word, relative_sub_word_list, hit_word_list)
        return result

    def get_hit_word_list(self, target_word, relative_sub_word_list):
        result_list = []
        for i in range(len(relative_sub_word_list)):
            sub_word = relative_sub_word_list[i]
            if target_word.find(sub_word) != -1:
                result_list.append(sub_word)
        return result_list

    def get_score(self, sub_word_index, target_word_index, model):
        in_embed = model.in_embed
        out_embed = model.out_embed
        # out_embed_weight = out_embed.weight.detach().numpy()
        sub_word_index = torch.tensor(sub_word_index, dtype=torch.long, device=model.device)
        target_word_index = torch.tensor(target_word_index, dtype=torch.long, device=model.device)
        sub_word_vector = in_embed(sub_word_index)
        # target_word_vector = out_embed(target_word_index)
        # dot = torch.dot(sub_word_vector, target_word_vector)
        score = out_embed(sub_word_vector)
        target_score = score[target_word_index]
        # print('sub_word_vector', sub_word_vector)
        # print('target_word_vector', target_word_vector)
        # print('dot', dot)
        # print('score', score)
        # print('target_score', target_score)
        return target_score.to('cpu').detach().numpy()

    def get_most_relative_sub_word(self, sub_word_index, sub_word, model, feature_vector_list, alternate_list):
        in_embed = model.in_embed
        sub_word_index = torch.tensor(sub_word_index, dtype=torch.long, device=model.device)
        sub_word_vector = in_embed(sub_word_index)
        # for i in range(len(feature_vector_list)):
        #     cur_vector = feature_vector_list[i]
        #     # print('cur_vector shape', cur_vector.shape)
        #     # print('sub_word_vector shape', sub_word_vector.shape)
        #     test_boolean = torch.eq(cur_vector, sub_word_vector)
        #     is_ok = True
        #     for j in range(len(test_boolean)):
        #         if test_boolean[j] == 0:
        #             is_ok = False
        #             break
        #     if is_ok:
        #         print('is_ok', i, cur_vector)
        cosine_similarity_list = torch.cosine_similarity(torch.Tensor(feature_vector_list),
                                                         sub_word_vector.to('cpu').unsqueeze(0).float(), dim=1)
        # print('feature_vector_list shape', feature_vector_list.shape)
        # print('sub_word_vector shape', sub_word_vector.shape)
        # print('cosine_similarity_list', cosine_similarity_list.shape, cosine_similarity_list)
        # test_cos = torch.cosine_similarity(sub_word_vector.to('cpu'),
        #                                    sub_word_vector.to('cpu'), dim=0)
        # print('test_cos', test_cos)
        value_list, index_list \
            = torch.topk(cosine_similarity_list, 10)
        # print('cosine_similarity_list', cosine_similarity_list)
        # print(value_list, index_list)
        result_list = []
        for i in range(len(index_list)):
            index = index_list[i]
            result_list.append(alternate_list[index])
            # print(sub_word, alternate_list[index], value_list[i])
        return result_list

    def build_feature_vector(self, model, embeddingIndexMapped):
        feature_vector_list = torch.Tensor([])
        alternate_list = []
        for word in embeddingIndexMapped.word_to_in_embedding_index:
            # subword
            if word not in embeddingIndexMapped.word_to_out_embedding_index and word != '<PADDING>':
                word_index = embeddingIndexMapped.word_to_in_embedding_index[word]
                word_index = torch.tensor(word_index, dtype=torch.long, device=model.device)
                feature_vector = model.in_embed(word_index)
                feature_vector = feature_vector.to('cpu')
                feature_vector = feature_vector.unsqueeze(0).float()
                feature_vector_list = torch.cat((
                    feature_vector_list,
                    feature_vector), dim=0)
                alternate_list.append(word)
        print('alternate_list size', len(alternate_list))
        print('shape', feature_vector_list.shape)
        return feature_vector_list, alternate_list


def single_main(dataStore, git_name, base_subword_version):
    print('single main', git_name, base_subword_version)
    detect = OOSVDetect()
    threshold = 3
    lowFreqContextsResultDao = LowFreqContextsResultDao()
    cbow_list = lowFreqContextsResultDao.query_by(git_name, base_subword_version + str(threshold))
    # cbowLowFreqTrace = LowFreqTrace(git_name, expected_validate_length, most_recent, threshold)
    my_list = []
    for doc in cbow_list:
        # 特定の条件に絞る
        predict_result = doc['predict_result']
        if int(predict_result['target']) != -1 and float(doc['new_rate']) >= 1:
            target = predict_result['target']
            top100_aq = predict_result['top100_aq']
            rank = get_rank(target, top100_aq)
            if 1 <= rank <= 5:
                my_list.append(doc)
    random.shuffle(my_list)
    print(len(my_list))
    max_count = 3
    count = 0
    result_list = []
    for i in range(len(my_list)):
        start_time = time.time()
        result = detect.detect_for_low_freq_trace(dataStore, my_list[i])
        result_list.append(result)
        print('cost time', time.time() - start_time, i)
        count += 1
        if count >= max_count:
            break
    save_result(result_list, git_name + '_analyse_result')


def save_result(data, file_name):
    project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    pkl_path_dir = project_dir + '/pkl/'
    pkl_path = pkl_path_dir + file_name + '.pkl'
    save_data(data, pkl_path)


def load_result(file_name):
    project_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    pkl_path_dir = project_dir + '/pkl/'
    pkl_path = pkl_path_dir + file_name + '.pkl'
    result = load_data(pkl_path)
    return result


def main(dataStore):
    git_name_list = ['tomcat', 'hadoop', 'lucene', 'hbase', 'cassandra', 'camel']
    # git_name_list = ['tomcat', 'lucene', 'hbase', 'cassandra']
    sub_version = '_1_20_paper_'
    for i in range(len(git_name_list)):
        git_name = git_name_list[i]
        subword_version_list = get_git_name_version(git_name, 'subword')
        base_subword_version = subword_version_list[0]['version'] + sub_version
        single_main(dataStore, git_name, base_subword_version)


if __name__ == '__main__':
    dataStore = DataStore()
    main(dataStore)
