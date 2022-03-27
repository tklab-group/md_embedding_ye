import sys

sys.path.append('../')
import torch
import time
from model.main import Main
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
    version = '2021_12_28_normal'
    mode = Mode.SUB_WORD
    max_epoch = 10
    dim = 700
    batch_size = 64
    lr = 1e-3
    is_negative_sampling = False
    is_sub_sampling = False
    is_subword_sub_sampling = False
    is_check_rename = True
    is_cosine_similarity_predict = False
    contribution_rate = 0.5
    is_use_package = True
    is_use_class_name = True
    is_use_return_type = True
    is_use_method_name = True
    is_use_param_type = True
    is_use_param_name = True
    is_split_train_data = False
    is_simple_handle_package = False
    is_simple_handle_class_name = False
    is_simple_handle_return_type = False
    is_simple_handle_method_name = False
    is_simple_handle_param_type = False
    is_simple_handle_param_name = False
    is_predict_with_file_level = False
    is_mark_respective_type = False
    is_only_new_file_context = False
    # check preprocessing
    is_preprocessing_package = True
    is_delete_modifier = True
    is_delete_void_return_type = True
    is_casing = False
    is_delete_single_subword = True
    is_delete_number_from_method_and_param = False
    is_number_type_token_from_return_and_param_type = False
    is_delete_sub_word_number = True
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
                                is_negative_sampling=is_negative_sampling,
                                is_sub_sampling=is_sub_sampling,
                                is_subword_sub_sampling=is_subword_sub_sampling,
                                is_check_rename=is_check_rename,
                                is_use_package=is_use_package,
                                is_use_class_name=is_use_class_name,
                                is_use_return_type=is_use_return_type,
                                is_use_method_name=is_use_method_name,
                                is_use_param_type=is_use_param_type,
                                is_use_param_name=is_use_param_name,
                                is_split_train_data=is_split_train_data,
                                is_simple_handle_package=is_simple_handle_package,
                                is_simple_handle_class_name=is_simple_handle_class_name,
                                is_simple_handle_return_type=is_simple_handle_return_type,
                                is_simple_handle_method_name=is_simple_handle_method_name,
                                is_simple_handle_param_type=is_simple_handle_param_type,
                                is_simple_handle_param_name=is_simple_handle_param_name,
                                is_predict_with_file_level=is_predict_with_file_level,
                                is_mark_respective_type=is_mark_respective_type,
                                is_preprocessing_package=is_preprocessing_package,
                                is_delete_modifier=is_delete_modifier,
                                is_delete_void_return_type=is_delete_void_return_type,
                                is_casing=is_casing,
                                is_delete_single_subword=is_delete_single_subword,
                                is_delete_number_from_method_and_param=is_delete_number_from_method_and_param,
                                is_number_type_token_from_return_and_param_type=is_number_type_token_from_return_and_param_type,
                                is_delete_sub_word_number=is_delete_sub_word_number
                                )
        dataLoader.debug_info()
        print()


