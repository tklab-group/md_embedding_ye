import sys
sys.path.append('../../')
from data.data_divider import DataDivider
from data.util import get_module_data
# from tests.data.test_data import get_method_map, get_module_data

def is_same(list_1, list_2):
    len_1 = len(list_1)
    len_2 = len(list_2)
    if len_1 == 0 or len_2 == 0:
        return False
    if len_1 != len_2:
        return False
    for i in range(len(list_1)):
        if list_1[i] != list_2[i]:
            return False
    return True

if __name__ == '__main__':
    md_list = get_module_data('tomcat')
    # md_list的顺序是倒过来的，因为在ModuleFinerGitService里面的实现，第一个md_list是最新的commit
    # 所以如果想实现train data和validate data从旧到新的话，那就需要反过来处理
    # md_list = []
    # total = 1000
    # for i in range(total):
    #     reverse_index = total - i
    #     md_list.append({
    #         'list': [reverse_index, reverse_index],
    #         'commitHash': str(reverse_index),
    #     })
    # print('md list size', len(md_list))
    # print('md_list', md_list)
    # dataDivider = DataDivider(md_list, 1000)
    dataDivider = DataDivider(md_list, 19)
    filter_condition_count = dataDivider.filter_condition_count()
    # print(filter_condition_count)
    train_data = dataDivider.get_train_data()
    train_data_commit_hash_list = dataDivider.train_data_commit_hash_list
    # print('train_data', len(train_data), train_data)
    # print('train_data_commit_hash_list')
    # for i in range(len(train_data_commit_hash_list)):
    #     print(train_data_commit_hash_list[i])
    validate_data = dataDivider.get_validate_data()
    validate_data_commit_hash_list = dataDivider.validate_data_commit_hash_list
    # print('validate_data', len(validate_data), validate_data)
    # print('validate_data_commit_hash_list')
    # for i in range(len(validate_data_commit_hash_list)):
    #     print(validate_data_commit_hash_list[i])
    for i in range(len(train_data_commit_hash_list)):
        cur_hash = train_data_commit_hash_list[i]
        for j in range(len(train_data_commit_hash_list)):
            target_hash = train_data_commit_hash_list[j]
            if i != j and cur_hash == target_hash:
                print('train error', i, j)

    for i in range(len(validate_data_commit_hash_list)):
        cur_hash = validate_data_commit_hash_list[i]
        for j in range(len(validate_data_commit_hash_list)):
            target_hash = validate_data_commit_hash_list[j]
            if i != j and cur_hash == target_hash:
                print('validate error', i, j)

    for i in range(len(train_data)):
        transaction = train_data[i]
        for j in range(len(train_data)):
            target_transaction = train_data[j]
            if i < j and i != j and is_same(transaction, target_transaction):
                print('train data error', i, j)

    for i in range(len(validate_data)):
        transaction = validate_data[i]
        for j in range(len(validate_data)):
            target_transaction = validate_data[j]
            if i < j and i != j and is_same(transaction, target_transaction):
                print('train data error', i, j)
