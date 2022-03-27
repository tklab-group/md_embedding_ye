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
from data.delete_record import DeleteRecord


if __name__ == '__main__':
    is_can_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_can_cuda else "cpu")
    print('is_can_cuda => ', is_can_cuda)
    if len(sys.argv) > 1:
        git_name = sys.argv[1]
        expected_validate_length = int(sys.argv[2])
        most_recent = int(sys.argv[3])
        repo_data = [{
            'git_name': git_name,
            'expected_validate_length': expected_validate_length,
            'most_recent': most_recent
        }]
        seed_str_list = sys.argv[4].split(',')
        seed_list = []
        for i in range(len(seed_str_list)):
            seed_list.append(int(seed_str_list[i]))
        version = sys.argv[5]
        max_epoch = int(sys.argv[6])
        dim = int(sys.argv[7])
        batch_size = int(sys.argv[8])
        mode_str = sys.argv[9]
        mode = Mode.NORMAL
        if mode_str == 'NORMAL':
            mode = Mode.NORMAL
        elif mode_str == 'SUB_WORD':
            mode = Mode.SUB_WORD
        elif mode_str == 'SUB_WORD_NO_FULL':
            mode = Mode.SUB_WORD_NO_FULL
        else:
            mode = Mode.N_GRAM
        is_casing_int = int(sys.argv[10])
        if is_casing_int == 0:
            is_casing = False
        else:
            is_casing = True
        is_preprocessing_package_int = int(sys.argv[11])
        if is_preprocessing_package_int == 0:
            is_preprocessing_package = False
        else:
            is_preprocessing_package = True
        device_id = int(sys.argv[12])
    else:
        repo_data = get_repo_data()
        seed_list = [1, 2]
        version = '2022_1_12_normal'
        max_epoch = 10
        dim = 700
        batch_size = 256
        mode = Mode.NORMAL
        mode_str = 'NORMAL'
        is_casing = True
        is_preprocessing_package = True
        device_id = 0
    if is_can_cuda:
        torch.cuda.set_device(device_id)
        print(torch.cuda.current_stream())
    print('repo_data', repo_data,
          'seed_list', seed_list,
          'version', version,
          'max_epoch', max_epoch,
          'dim', dim,
          'batch_size', batch_size,
          'mode_str', mode_str,
          'is_casing', is_casing,
          'is_preprocessing_package', is_preprocessing_package,
          'device_id', device_id)
    seed = 6
    # subword 20 700 128 is_preprocessing_package=True is_casing=True 63.07s 63.07s*1000/3600 = 17.5h
    # normal 10 700 256 19.6s 19.6s*1000/3600 = 5.44h
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
    # is_preprocessing_package = True
    is_delete_modifier = True
    is_delete_void_return_type = True
    # is_casing = True
    is_delete_single_subword = True
    is_delete_number_from_method_and_param = False
    is_number_type_token_from_return_and_param_type = False
    is_delete_sub_word_number = True
    shuffle = False
    print('mode, max_epoch, dim, batch_size => ', mode, max_epoch, dim, batch_size)

    dataStore = DataStore()

    for i in range(len(repo_data)):
        repo_data_item = repo_data[i]
        git_name = repo_data_item['git_name']
        expected_validate_length = repo_data_item['expected_validate_length']
        most_recent = repo_data_item['most_recent']
        deleteRecord = DeleteRecord(git_name, expected_validate_length)
        print('git_name, expected_validate_length => ', git_name, expected_validate_length)
        dataLoader = DataLoader(git_name=git_name,
                                expected_validate_length=expected_validate_length,
                                most_recent=most_recent,
                                dataStore=dataStore,
                                deleteRecord=deleteRecord,
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
        print('fix transaction:')
        main = Main(git_name=git_name,
                    expected_validate_length=expected_validate_length,
                    most_recent=most_recent,
                    mode=mode,
                    max_epoch=max_epoch,
                    dim=dim,
                    batch_size=batch_size,
                    lr=lr,
                    dataStore=dataStore,
                    deleteRecord=deleteRecord,
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
                    is_delete_sub_word_number=is_delete_sub_word_number,
                    seed=seed,
                    shuffle=shuffle
                    )
        train_start_time = time.time()
        main.train(device=device)
        print('train time:', time.time() - train_start_time)
        validate_start_time = time.time()
        main.eval(
            is_mix=False,
            is_only_new_file_context=is_only_new_file_context
        )
        print('validate time:', time.time() - validate_start_time)
        del main.model
        torch.cuda.empty_cache()
        print('no fix transaction:')
        for j in range(len(seed_list)):
            seed = seed_list[j]
            cur_version = version + '_' + git_name + '_' + str(seed)
            print('cur_version', cur_version)
            multiMain = MultiMain(dim=dim,
                                  batch_size=batch_size,
                                  max_epoch=max_epoch,
                                  git_name=git_name,
                                  expected_validate_length=expected_validate_length,
                                  most_recent=most_recent,
                                  mode=mode,
                                  validate_data=dataLoader.dataDivider.get_validate_data(),
                                  device=device,
                                  dataStore=dataStore,
                                  deleteRecord=deleteRecord,
                                  is_negative_sampling=is_negative_sampling,
                                  is_sub_sampling=is_sub_sampling,
                                  is_check_rename=is_check_rename,
                                  is_cosine_similarity_predict=is_cosine_similarity_predict,
                                  contribution_rate=contribution_rate,
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
                                  version=cur_version,
                                  seed=seed,
                                  shuffle=shuffle
                                  )
            multiMain.train_and_eval()
            multiMain.summary()

            print('-------------------------------------------------------', 'end with seed:', seed)
            print()

