import time
import datetime
# from model.model import Model
# from model.model import save_model
# my_model = Model(name="test")
# my_model.save()
# save_model(None)
# list = range(1)
# print(list)
# for i in list:
#     print(i)

hash_table = {}
f = 100
v_list = []
for i in range(10):
    v = [i for z in range(f)]
    t = tuple(v)
    hash_table[t] = i

test_v = [2 for z in range(f)]
result = hash_table[tuple(test_v)]
print(result)

test_a = [0, 1, 2, 3, 4]
print(test_a[0: 3])

now_time = int(time.time())
print(datetime.datetime.fromtimestamp(now_time))

str1 = '[EVENT_TYPE-extends-Event[#]]_void_validateEvent(EVENT_TYPE_event)'
str2 = '[EVENT_TYPE-extends-Event[#]]'
str3 = str1.lstrip(str2)
str4 = str1.split(str2, 1)[1]
print(str3, str4)
