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
from data.mongo import PreprocessResultDao
import datetime

if __name__ == '__main__':
    is_can_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_can_cuda else "cpu")
    print('is_can_cuda => ', is_can_cuda)
    repo_data = get_repo_data()
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
    is_preprocessing_package_list = [True, False]
    is_delete_modifier_list = [True]
    is_delete_void_return_type_list = [True]
    is_casing_list = [True, False]
    is_delete_single_subword_list = [True]
    is_delete_number_from_method_and_param_list = [False]
    is_number_type_token_from_return_and_param_type_list = [False]
    seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    version = '2021-12-27-preprocess-'
    print('mode, max_epoch, dim, batch_size => ', mode, max_epoch, dim, batch_size)

    dataStore = DataStore()

    dao = PreprocessResultDao()
    total_loop = 2 ** 7
    print('start with', datetime.datetime.now(), total_loop)
    for i in range(len(repo_data)):
        repo_data_item = repo_data[i]
        git_name = repo_data_item['git_name']
        expected_validate_length = repo_data_item['expected_validate_length']
        most_recent = repo_data_item['most_recent']
        # print('git_name, expected_validate_length => ', git_name, expected_validate_length)
        for i1 in range(len(is_preprocessing_package_list)):
            for i2 in range(len(is_delete_modifier_list)):
                for i3 in range(len(is_delete_void_return_type_list)):
                    for i4 in range(len(is_casing_list)):
                        for i5 in range(len(is_delete_single_subword_list)):
                            for i6 in range(len(is_delete_number_from_method_and_param_list)):
                                for i7 in range(len(is_number_type_token_from_return_and_param_type_list)):
                                    for i8 in range(len(seed_list)):
                                        seed = seed_list[i8]
                                        is_preprocessing_package = is_preprocessing_package_list[i1]
                                        is_delete_modifier = is_delete_modifier_list[i2]
                                        is_delete_void_return_type = is_delete_void_return_type_list[i3]
                                        is_casing = is_casing_list[i4]
                                        is_delete_single_subword = is_delete_single_subword_list[i5]
                                        is_delete_number_from_method_and_param = \
                                        is_delete_number_from_method_and_param_list[i6]
                                        is_number_type_token_from_return_and_param_type = \
                                        is_number_type_token_from_return_and_param_type_list[i7]
                                        print(is_preprocessing_package, is_delete_modifier, is_delete_void_return_type,
                                              is_casing, is_delete_single_subword,
                                              is_delete_number_from_method_and_param,
                                              is_number_type_token_from_return_and_param_type)
                                        main = Main(git_name=git_name,
                                                    expected_validate_length=expected_validate_length,
                                                    most_recent=most_recent,
                                                    mode=mode,
                                                    max_epoch=max_epoch,
                                                    dim=dim,
                                                    batch_size=batch_size,
                                                    lr=lr,
                                                    dataStore=dataStore,
                                                    is_negative_sampling=is_negative_sampling,
                                                    is_subword_sub_sampling=is_subword_sub_sampling,
                                                    is_sub_sampling=is_sub_sampling,
                                                    is_check_rename=is_check_rename,
                                                    is_cosine_similarity_predict=is_cosine_similarity_predict,
                                                    contribution_rate=contribution_rate,
                                                    is_fix=True,
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
                                                    seed=seed
                                                    )
                                        train_start_time = time.time()
                                        main.train(device=device)
                                        train_cost_time = time.time() - train_start_time
                                        print('train time:', time.time() - train_start_time)
                                        validate_start_time = time.time()
                                        param_recall_result = main.eval(
                                            is_mix=False,
                                            is_only_new_file_context=is_only_new_file_context
                                        )
                                        validate_cost_time = time.time() - validate_start_time
                                        print('validate time:', time.time() - validate_start_time)
                                        str_mode = 'NORMAL'
                                        if mode == Mode.NORMAL:
                                            str_mode = 'NORMAL'
                                        elif mode == Mode.SUB_WORD:
                                            str_mode = 'SUB_WORD'
                                        elif mode == Mode.SUB_WORD_NO_FULL:
                                            str_mode = 'SUB_WORD_NO_FULL'
                                        else:
                                            str_mode = 'N_GRAM'
                                        dao.insert({
                                            'git_name': git_name,
                                            'expected_validate_length': expected_validate_length,
                                            'most_recent': most_recent,
                                            'mode': str_mode,
                                            'max_epoch': max_epoch,
                                            'dim': dim,
                                            'batch_size': batch_size,
                                            'lr': lr,
                                            'is_fix': True,
                                            'train_time': train_cost_time,
                                            'validate_time': validate_cost_time,
                                            'recall_list': param_recall_result,
                                            'version': version + str(seed),
                                            "last_modified": datetime.datetime.utcnow(),
                                            'is_preprocessing_package': is_preprocessing_package,
                                            'is_delete_modifier': is_delete_modifier,
                                            'is_delete_void_return_type': is_delete_void_return_type,
                                            'is_casing': is_casing,
                                            'is_delete_single_subword': is_delete_single_subword,
                                            'is_delete_number_from_method_and_param': is_delete_number_from_method_and_param,
                                            'is_number_type_token_from_return_and_param_type': is_number_type_token_from_return_and_param_type,
                                            'is_delete_sub_word_number': True,
                                            'seed': seed
                                        })
                                        del main.model
                                        torch.cuda.empty_cache()
