import sys
sys.path.append('../')
import torch
import time
import datetime
from data.data_loader import DataLoader
from model.trainer import Trainer
from model.cbow import EmbeddingModel
from model.cbow_negative_sampling import EmbeddingModel as EmbeddingModelNegativeSampling
from model.extend_cbow import ExtendEmbeddingModel
from model.eval import Evaluation
from model.mix_eval import MixEvaluation
from model.ranking_eval import RankingEvaluation
from data.mode_enum import Mode
from data.data_store import DataStore
from data.delete_record import DeleteRecord


def print_recall(k, is_consider_new_file, is_predict_new_file, micro_recall, macro_recall):
    # print('k %d | new file %d | predict new file %d | micro recall %.2f | macro recall %.2f'
    #       % (k, is_consider_new_file, is_predict_new_file, micro_recall, macro_recall))
    print('k %d | new file %d | micro recall %.2f | macro recall %.2f'
          % (k, is_consider_new_file, micro_recall, macro_recall))
    # print('k %d | micro recall %.2f | macro recall %.2f'
    #       % (k, micro_recall, macro_recall))


class Main:
    def __init__(self,
                 dataStore: DataStore,
                 deleteRecord: DeleteRecord,
                 dim=100,
                 batch_size=32,
                 max_epoch=10,
                 git_name=None,
                 expected_validate_length=1000,
                 most_recent=5000,
                 is_test=False,
                 test_md_list=None,
                 test_method_map=None,
                 mode=Mode.NORMAL,
                 lr=1e-3,
                 is_sub_sampling=False,
                 is_subword_sub_sampling=False,
                 is_negative_sampling=False,
                 is_cosine_similarity_predict=False,
                 is_check_rename=True,
                 is_contexts_extend=False,
                 contribution_rate=0.5,
                 is_fix=True,
                 is_print_info=True,
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
                 seed=6,
                 shuffle=False
                 ):
        self.dim = dim
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.git_name = git_name
        self.expected_validate_length = expected_validate_length
        self.most_recent = most_recent
        self.data_loader = None
        self.model = None
        self.mode = mode
        self.lr = lr
        self.is_sub_sampling = is_sub_sampling
        self.is_subword_sub_sampling = is_subword_sub_sampling
        self.is_negative_sampling = is_negative_sampling
        self.is_cosine_similarity_predict = is_cosine_similarity_predict
        self.is_check_rename = is_check_rename
        self.is_contexts_extend = is_contexts_extend
        self.contribution_rate = contribution_rate
        self.is_fix = is_fix
        self.is_print_info = is_print_info
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
        self.seed = seed
        self.shuffle = shuffle

        self.is_test = is_test
        self.test_md_list = test_md_list
        self.test_method_map = test_method_map

        self.dataStore = dataStore
        self.deleteRecord = deleteRecord

    def train(self, is_load_model=False, load_path=None, device=None, is_load_from_pkl=False):
        data_loader = DataLoader(git_name=self.git_name,
                                 expected_validate_length=self.expected_validate_length,
                                 most_recent=self.most_recent,
                                 is_load_from_pkl=is_load_from_pkl,
                                 mode=self.mode,
                                 is_test=self.is_test,
                                 test_md_list=self.test_md_list,
                                 test_method_map=self.test_method_map,
                                 is_sub_sampling=self.is_sub_sampling,
                                 is_subword_sub_sampling=self.is_subword_sub_sampling,
                                 is_negative_sampling=self.is_negative_sampling,
                                 dataStore=self.dataStore,
                                 deleteRecord=self.deleteRecord,
                                 is_check_rename=self.is_check_rename,
                                 is_contexts_extend=self.is_contexts_extend,
                                 is_use_package=self.is_use_package,
                                 is_use_class_name=self.is_use_class_name,
                                 is_use_return_type=self.is_use_return_type,
                                 is_use_method_name=self.is_use_method_name,
                                 is_use_param_type=self.is_use_param_type,
                                 is_use_param_name=self.is_use_param_name,
                                 is_split_train_data=self.is_split_train_data,
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
        # data_loader.debug_info()
        in_embedding_num = data_loader.embeddingIndexMapped.in_embedding_num
        out_embedding_num = data_loader.embeddingIndexMapped.out_embedding_num
        # print('in_embedding_num', in_embedding_num)
        train_contexts_target = data_loader.train_contexts_target
        # print('train_contexts_target', train_contexts_target)
        contexts = train_contexts_target['contexts']
        target = train_contexts_target['target']
        negative_sampling = train_contexts_target['negative_sampling']

        model = None
        if not self.is_negative_sampling:
            model = EmbeddingModel(in_embedding_num, out_embedding_num, self.dim, device, seed=self.seed).to(device)
            # test
            # model = ExtendEmbeddingModel(in_embedding_num, out_embedding_num, self.dim, device).to(device)
        else:
            model = EmbeddingModelNegativeSampling(
                in_embedding_num, out_embedding_num, self.dim, device, seed=self.seed).to(device)
        if is_load_model:
            model.load_state_dict(torch.load(load_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        self.data_loader = data_loader
        self.model = model

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=device,
            is_fix=self.is_fix,
            is_print_info=self.is_print_info,
            shuffle=self.shuffle)
        trainer.fit(contexts=contexts,
                    target=target,
                    batch_size=self.batch_size,
                    max_epoch=self.max_epoch,
                    is_negative_sampling=self.is_negative_sampling,
                    negative_sampling=negative_sampling)
        # if not self.is_test:
        #     # save model
        #     model_signal = 'normal'
        #     if self.mode == Mode.SUB_WORD:
        #         model_signal = 'sub_word'
        #     if self.mode == Mode.SUB_WORD_NO_FULL:
        #         model_signal = 'sub_word_no_full'
        #     if self.mode == Mode.N_GRAM:
        #         model_signal = 'n_gram'
        #     time_str = str(datetime.datetime.fromtimestamp(int(time.time())))
        #     param_info = 'd' + str(self.dim) + '_bs' + str(self.batch_size) \
        #                       + '_me' + str(self.max_epoch) + '_evl' + str(self.expected_validate_length) + '_'
        #     save_path = '../trained_model_params/' + model_signal + '_' + self.git_name + '_' + param_info \
        #                 + time_str + '.pth'
        #     print('save_path', save_path)
        #     torch.save(model.state_dict(), save_path)
        #     trainer.plot()

    def eval(self, is_mix=False, is_only_new_file_context=False, is_consider_new_file=True):
        self.model.eval()
        if is_mix and not self.is_predict_with_file_level:
            mixEvaluation = MixEvaluation(
                model=self.model,
                data_loader=self.data_loader,
                mode=self.mode,
                is_negative_sampling=self.is_negative_sampling,
                is_cosine_similarity_predict=self.is_cosine_similarity_predict,
                is_fix_transaction=True,
                contribution_rate=self.contribution_rate
                )
            mix_metric_list = mixEvaluation.validate(False)

        evaluation = Evaluation(
            model=self.model,
            data_loader=self.data_loader,
            mode=self.mode,
            is_negative_sampling=self.is_negative_sampling,
            is_cosine_similarity_predict=self.is_cosine_similarity_predict,
            is_split=self.is_split_train_data,
            is_predict_with_file_level=self.is_predict_with_file_level
        )
        metric_list = evaluation.validate(
            is_predict_new_file=False,
            is_only_new_file_context=is_only_new_file_context
        )
        # metric_list = mixEvaluation.validate(False)

        # evaluationNew = Evaluation(
        #     model=self.model,
        #     data_loader=self.data_loader,
        #     mode=self.mode,
        #     is_negative_sampling=self.is_negative_sampling,
        #     is_cosine_similarity_predict=self.is_cosine_similarity_predict)
        # metric_list_new = evaluationNew.validate(True)
        if is_mix and not self.is_predict_with_file_level:
            if self.is_print_info:
                print('only cbow eval:')
        param_recall_result = []
        for i in range(len(metric_list)):
            k = evaluation.k_list[i]

            metric = metric_list[k]
            # is_consider_new_file=False
            micro_recall, macro_recall = metric.summary(is_consider_new_file)
            if self.is_print_info:
                print_recall(k, is_consider_new_file, False, micro_recall, macro_recall)
            param_recall_result.append({
                'k': k,
                'micro_recall': micro_recall,
                'macro_recall': macro_recall
            })
            # is_consider_new_file=True
            # micro_recall, macro_recall = metric.summary(True)
            # print_recall(k, True, False, micro_recall, macro_recall)

            # metricNew = metric_list_new[k]
            # # is_consider_new_file=False
            # micro_recall, macro_recall = metricNew.summary(False)
            # print_recall(k, False, True, micro_recall, macro_recall)
            # # is_consider_new_file=True
            # micro_recall, macro_recall = metricNew.summary(True)
            # print_recall(k, True, True, micro_recall, macro_recall)

        if is_mix and not self.is_predict_with_file_level:
            if self.is_print_info:
                print('mix eval:')
            for i in range(len(mix_metric_list)):
                k = mixEvaluation.k_list[i]

                metric = mix_metric_list[k]
                # is_consider_new_file=False
                micro_recall, macro_recall = metric.summary(False)
                print_recall(k, False, False, micro_recall, macro_recall)
                # is_consider_new_file=True
                # micro_recall, macro_recall = metric.summary(True)
                # print_recall(k, True, False, micro_recall, macro_recall)
        return param_recall_result
