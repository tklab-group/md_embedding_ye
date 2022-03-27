from model.predict import Predict


# l = [11, 12, 13, 14, 15]
# predict.create_query(l)
# predict.create_query_expected()
# predict.print_pair_list()
# print(predict.check(l))

# l = ['a', 'b', 'c', 'd', 'e']
l = [100, 200]
for i in range(10):
    l.append(i)
    predict = Predict()
    predict.create_query_leave_one_out(l)
    predict.create_query_expected_leave_one_out(l)
    # predict.print_pair_list()
    print(len(predict.pair_list), len(l))
    # print(predict.check(l))

