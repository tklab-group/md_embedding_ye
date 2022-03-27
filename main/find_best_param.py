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
from data.mongo import ParamRecallDao
import datetime
from data.delete_record import DeleteRecord

if __name__ == '__main__':
    is_can_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_can_cuda else "cpu")
    print('is_can_cuda => ', is_can_cuda)
    repo_data = get_repo_data()
    mode_list = [Mode.SUB_WORD]
    # 2*5*4*2*2*2
    max_epoch_list = [10, 20]
    dim_list = [300, 400, 500, 600, 700]
    batch_size_list = [64, 128, 256, 512, 1024]
    lr_list = [1e-3]
    shuffle_list = [False]
    is_preprocessing_package_list = [True, False]
    is_casing_list = [True, False]
    version = '2022-1-6-test-l'
    contribution_rate = 0.5
    dataStore = DataStore()

    dao = ParamRecallDao()
    total_loop = len(mode_list) * len(max_epoch_list) * len(dim_list) \
                 * len(batch_size_list) * len(lr_list) \
                 * len(shuffle_list) * len(is_preprocessing_package_list) * len(is_casing_list)
    print('start with', datetime.datetime.now(), total_loop)
    for i in range(len(repo_data)):
        repo_data_item = repo_data[i]
        git_name = repo_data_item['git_name']
        expected_validate_length = repo_data_item['expected_validate_length']
        most_recent = repo_data_item['most_recent']
        deleteRecord = DeleteRecord(git_name, expected_validate_length, most_recent)
        # print('git_name, expected_validate_length => ', git_name, expected_validate_length)
        for i1 in range(len(mode_list)):
            for i2 in range(len(max_epoch_list)):
                for i3 in range(len(dim_list)):
                    for i4 in range(len(batch_size_list)):
                        for i5 in range(len(lr_list)):
                            for i6 in range(len(shuffle_list)):
                                for i7 in range(len(is_preprocessing_package_list)):
                                    for i8 in range(len(is_casing_list)):
                                        mode = mode_list[i1]
                                        max_epoch = max_epoch_list[i2]
                                        dim = dim_list[i3]
                                        batch_size = batch_size_list[i4]
                                        lr = lr_list[i5]
                                        shuffle = shuffle_list[i6]
                                        is_preprocessing_package = is_preprocessing_package_list[i7]
                                        is_casing = is_casing_list[i8]
                                        print('git name, mode, max_epoch, dim, batch_size, lr shuffle, '
                                              'is_preprocessing_package, is_casing=> ',
                                              git_name,
                                              mode,
                                              max_epoch,
                                              dim,
                                              batch_size,
                                              lr,
                                              shuffle,
                                              is_preprocessing_package,
                                              is_casing)
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
                                                    contribution_rate=contribution_rate,
                                                    is_fix=True,
                                                    is_print_info=False,
                                                    is_preprocessing_package=is_preprocessing_package,
                                                    is_casing=is_casing,
                                                    shuffle=shuffle,
                                                    )
                                        train_start_time = time.time()
                                        main.train(device=device)
                                        train_cost_time = time.time() - train_start_time
                                        # print('train time:', train_cost_time)
                                        validate_start_time = time.time()
                                        param_recall_result = main.eval(
                                            is_consider_new_file=False
                                        )
                                        param_recall_result_new = main.eval(
                                            is_consider_new_file=True
                                        )
                                        validate_cost_time = time.time() - validate_start_time
                                        # print('validate time:', validate_cost_time)
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
                                            'recall_list_new': param_recall_result_new,
                                            'version': version,
                                            "last_modified": datetime.datetime.utcnow(),
                                            'shuffle': shuffle,
                                            'is_preprocessing_package': is_preprocessing_package,
                                            'is_casing': is_casing
                                        })
                                        print()
