import sys
sys.path.append('../../')
import time
import datetime
from data.data_loader import DataLoader
from data.mode_enum import Mode
from data.id_mapped import IdMapped
from data.younger_trace import YoungerTrace

all_id_to_word = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

youngerTrace = YoungerTrace(all_id_to_word)

test_md_id_list = [1, 2, 3, 4, 5, 6]

vocab_list_1 = ['A', 'B']
transaction_1 = [1, 2, 6]

# 先评价，后trace
print(youngerTrace.freq)
for i in range(len(test_md_id_list)):
    md_id = test_md_id_list[i]
    print(md_id, youngerTrace.is_younger_md_id(md_id))
youngerTrace.trace(vocab_list_1, transaction_1)


vocab_list_2 = ['B', 'C']
transaction_2 = [1, 2, 3, 6]
# 先评价，后trace
print(youngerTrace.freq)
for i in range(len(test_md_id_list)):
    md_id = test_md_id_list[i]
    print(md_id, youngerTrace.is_younger_md_id(md_id))
youngerTrace.trace(vocab_list_1, transaction_1)
# youngerTrace.trace(vocab_list_2, transaction_2)
# youngerTrace.trace(vocab_list_2, transaction_2)
# youngerTrace.trace(vocab_list_2, transaction_2)

# print(youngerTrace.freq)
# for i in range(len(test_md_id_list)):
#     md_id = test_md_id_list[i]
#     print(md_id, youngerTrace.is_younger_md_id(md_id))


