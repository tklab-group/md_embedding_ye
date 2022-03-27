import sys
sys.path.append('../../')
from data.data_divider import DataDivider
from data.util import get_module_data
# from tests.data.test_data import get_method_map, get_module_data


# md_list = get_module_data()
md_list = get_module_data('tomcat')
print('md list size', len(md_list))
# dataDivider = DataDivider(md_list, 1000)
dataDivider = DataDivider(md_list, 1000)
filter_condition_count = dataDivider.filter_condition_count()
print(filter_condition_count)
train_data = dataDivider.get_train_data()
validate_data = dataDivider.get_validate_data()
print('train data size', len(train_data))
print('validate data size', len(validate_data))
dataDivider.count_average_num()
# print('train data size', len(train_data), train_data)
# print('train data size', len(train_data), train_data[100], train_data[900])
# print('validate data size', len(validate_data), validate_data)
# print('validate data size', len(validate_data), validate_data[300], validate_data[500])
# filter_count = 21
# validate data
# print('validate data index: ')
# for i in range(filter_count - 1, -1, -1):
#     print(i)
# print('train data index: ')
# for i in range(len(md_list) - filter_count - 1, -1, -1):
#     print(i)
