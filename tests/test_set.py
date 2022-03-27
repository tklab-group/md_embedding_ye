test_set = set()
test_set.add(100)
test_set.add(2)
test_set.add(3)
test_set.add(4)
test_list = list(test_set)
for i in range(len(test_list)):
    print(test_list[i])
print(max(test_list))
print(max(test_set))

my_list = [1, 2, 3, 4, 4, 1]
my_list2 = list(set(my_list))
print(my_list2)
