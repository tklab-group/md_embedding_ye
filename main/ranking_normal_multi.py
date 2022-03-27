import sys

sys.path.append('../')
import torch
import time
from model.main import Main
from model.ranking_main import RankingMain
from model.multi_main import MultiMain
from data.mode_enum import Mode
from data.repo_data import get_repo_data
from data.data_loader import DataLoader
from data.data_store import DataStore

if __name__ == '__main__':
    is_can_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_can_cuda else "cpu")
    print('is_can_cuda => ', is_can_cuda)
    repo_data = get_repo_data()
    mode = Mode.SUB_WORD
    max_epoch = 10
    dim = 700
    batch_size = 512
    lr = 1e-3
    is_use_package = True
    is_use_class_name = True
    is_use_return_type = True
    is_use_method_name = True
    is_use_param_type = True
    is_use_param_name = True
    is_simple_handle_package = False
    is_simple_handle_class_name = False
    is_simple_handle_return_type = False
    is_simple_handle_method_name = False
    is_simple_handle_param_type = False
    is_simple_handle_param_name = False
    pre_ranking_top_k = 20
    pre_ranking_file_level_top_k = 20
    print('mode, max_epoch, dim, batch_size => ', mode, max_epoch, dim, batch_size)

    dataStore = DataStore()

    for i in range(len(repo_data)):
        repo_data_item = repo_data[i]
        git_name = repo_data_item['git_name']
        expected_validate_length = repo_data_item['expected_validate_length']
        most_recent = repo_data_item['most_recent']
        print('git_name, expected_validate_length => ', git_name, expected_validate_length)
        dataLoader = DataLoader(git_name=git_name,
                                expected_validate_length=expected_validate_length,
                                most_recent=most_recent,
                                dataStore=dataStore,
                                mode=mode,
                                is_negative_sampling=False,
                                is_sub_sampling=False,
                                is_check_rename=True,
                                is_use_package=is_use_package,
                                is_use_class_name=is_use_class_name,
                                is_use_return_type=is_use_return_type,
                                is_use_method_name=is_use_method_name,
                                is_use_param_type=is_use_param_type,
                                is_use_param_name=is_use_param_name,
                                is_split_train_data=False,
                                is_simple_handle_package=is_simple_handle_package,
                                is_simple_handle_class_name=is_simple_handle_class_name,
                                is_simple_handle_return_type=is_simple_handle_return_type,
                                is_simple_handle_method_name=is_simple_handle_method_name,
                                is_simple_handle_param_type=is_simple_handle_param_type,
                                is_simple_handle_param_name=is_simple_handle_param_name,
                                is_predict_with_file_level=True
                                )
        dataLoader.debug_info()
        print('fix transaction:')
        main = RankingMain(git_name=git_name,
                           expected_validate_length=expected_validate_length,
                           most_recent=most_recent,
                           mode=mode,
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
                           pre_ranking_top_k=pre_ranking_top_k,
                           pre_ranking_file_level_top_k=pre_ranking_file_level_top_k
                           )
        train_start_time = time.time()
        main.train()
        print('train time:', time.time() - train_start_time)
        validate_start_time = time.time()
        main.eval()
        print('validate time:', time.time() - validate_start_time)
        del main.main.model
        del main.mainFileLevel.model
        torch.cuda.empty_cache()
        print('-------------------------------------------------------')
        print()
