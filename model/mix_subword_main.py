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
from model.mix_eval_subword import MixEvaluationSubword
from data.mode_enum import Mode
from data.data_store import DataStore
from model.main import Main


def print_recall(k, micro_recall, macro_recall):
    print('k %d | micro recall %.2f | macro recall %.2f'
          % (k, micro_recall, macro_recall))


class MixSubwordMain:
    def __init__(self,
                 dataStore: DataStore,
                 dim=700,
                 # normal
                 batch_size=512,
                 max_epoch=20,
                 # subword param-------
                 batch_size_subword=64,
                 max_epoch_subword=10,
                 git_name=None,
                 expected_validate_length=1000,
                 most_recent=5000,
                 mode=Mode.SUB_WORD,
                 lr=1e-3,
                 is_use_package=True,
                 is_use_class_name=True,
                 is_use_return_type=True,
                 is_use_method_name=True,
                 is_use_param_type=True,
                 is_use_param_name=True,
                 is_simple_handle_package=False,
                 is_simple_handle_class_name=False,
                 is_simple_handle_return_type=False,
                 is_simple_handle_method_name=False,
                 is_simple_handle_param_type=False,
                 is_simple_handle_param_name=False,
                 is_subword_sub_sampling=False,
                 ):
        self.mode = mode
        is_can_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if is_can_cuda else "cpu")
        self.main = Main(git_name=git_name,
                         expected_validate_length=expected_validate_length,
                         most_recent=most_recent,
                         mode=Mode.NORMAL,
                         max_epoch=max_epoch,
                         dim=dim,
                         batch_size=batch_size,
                         lr=lr,
                         dataStore=dataStore,
                         is_use_package=is_use_package,
                         is_use_class_name=is_use_class_name,
                         is_use_return_type=is_use_return_type,
                         is_use_method_name=is_use_method_name,
                         is_use_param_type=is_use_param_type,
                         is_use_param_name=is_use_param_name,
                         is_simple_handle_package=is_simple_handle_package,
                         is_simple_handle_class_name=is_simple_handle_class_name,
                         is_simple_handle_return_type=is_simple_handle_return_type,
                         is_simple_handle_method_name=is_simple_handle_method_name,
                         is_simple_handle_param_type=is_simple_handle_param_type,
                         is_simple_handle_param_name=is_simple_handle_param_name,
                         is_predict_with_file_level=False
                         )
        self.mainSubword = Main(git_name=git_name,
                                expected_validate_length=expected_validate_length,
                                most_recent=most_recent,
                                mode=mode,
                                max_epoch=max_epoch_subword,
                                dim=dim,
                                batch_size=batch_size_subword,
                                lr=lr,
                                dataStore=dataStore,
                                is_use_package=is_use_package,
                                is_use_class_name=is_use_class_name,
                                is_use_return_type=is_use_return_type,
                                is_use_method_name=is_use_method_name,
                                is_use_param_type=is_use_param_type,
                                is_use_param_name=is_use_param_name,
                                is_simple_handle_package=is_simple_handle_package,
                                is_simple_handle_class_name=is_simple_handle_class_name,
                                is_simple_handle_return_type=is_simple_handle_return_type,
                                is_simple_handle_method_name=is_simple_handle_method_name,
                                is_simple_handle_param_type=is_simple_handle_param_type,
                                is_simple_handle_param_name=is_simple_handle_param_name,
                                is_predict_with_file_level=False,
                                is_subword_sub_sampling=is_subword_sub_sampling
                                )

    def train(self):
        self.main.train(device=self.device)
        self.mainSubword.train(device=self.device)

    def eval(self):
        self.main.model.eval()
        self.mainSubword.model.eval()

        evaluation = MixEvaluationSubword(
            model_normal=self.main.model,
            data_loader_normal=self.main.data_loader,
            model_subword=self.mainSubword.model,
            data_loader_subword=self.mainSubword.data_loader,
        )
        metric_list = evaluation.validate()

        param_recall_result = []
        for i in range(len(metric_list)):
            k = evaluation.k_list[i]

            metric = metric_list[k]
            # is_consider_new_file=False
            micro_recall, macro_recall = metric.summary(False)
            print_recall(k, micro_recall, macro_recall)
            param_recall_result.append({
                'k': k,
                'micro_recall': micro_recall,
                'macro_recall': macro_recall
            })

        return param_recall_result
