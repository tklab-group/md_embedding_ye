import time
import sys
# sys.path.append('../../../')
# from data.mode_enum import Mode
if __name__ == '__main__':
    # sys.argv[0] = script name
    # sys.argv[1] = git_name
    # sys.argv[2] = expected_validate_length
    # sys.argv[3] = most_recent
    # sys.argv[4] = seed_list
    print(sys.argv)
    if len(sys.argv) > 0:
        git_name = sys.argv[1]
        expected_validate_length = sys.argv[2]
        most_recent = sys.argv[3]
        seed_str_list = sys.argv[4].split(',')
        seed_list = []
        for i in range(len(seed_str_list)):
            seed_list.append(int(seed_str_list[i]))
        version = sys.argv[5]
        max_epoch = int(sys.argv[6])
        dim = int(sys.argv[7])
        batch_size = int(sys.argv[8])
        mode_str = sys.argv[9]
        # mode = Mode.NORMAL
        # if mode_str == 'NORMAL':
        #     mode = Mode.NORMAL
        # elif mode_str == 'SUB_WORD':
        #     mode = Mode.SUB_WORD
        # elif mode_str == 'SUB_WORD_NO_FULL':
        #     mode = Mode.SUB_WORD_NO_FULL
        # else:
        #     mode = Mode.N_GRAM
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
        print('git_name', git_name,
              'expected_validate_length', expected_validate_length,
              'most_recent', most_recent,
              'seed_list', seed_list,
              'version', version,
              'max_epoch', max_epoch,
              'dim', dim,
              'batch_size', batch_size,
              'mode_str', mode_str,
              'is_casing', is_casing,
              'is_preprocessing_package', is_preprocessing_package,
              'device_id', device_id)
    time.sleep(2)
    print(1)

