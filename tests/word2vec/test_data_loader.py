from data.data_loader import DataLoader, test_experiment_data, create_leave_one_out_contexts_target

data_loader = DataLoader('LCExtractor', 19)
train_data = data_loader.train_data
validate_data = data_loader.validate_data
print('validate_data_num', data_loader.validate_data_num)
print('validate_end_count', data_loader.validate_end_count)
print('train', len(train_data['contexts']))
print('validate', len(validate_data['contexts']))
# test_experiment_data('LCExtractor')

# l = [100, 200]
# for i in range(10):
#     l.append(i)
#     result = create_leave_one_out_contexts_target(l)
#     # predict.print_pair_list()
#     print(len(result), len(l),  result)
#     # print(predict.check(l))
