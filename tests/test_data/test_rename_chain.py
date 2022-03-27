import sys

sys.path.append('../../')
import time
import datetime
from data.data_loader import DataLoader
from data.mode_enum import Mode
from data.data_store import DataStore

git_name = 'tomcat'
# git_name = 'LCExtractor'
expected_validate_length = 1000
most_recent = 5000
dataStore = DataStore()
data_loader = DataLoader(git_name=git_name,
                         expected_validate_length=expected_validate_length,
                         most_recent=most_recent,
                         mode=Mode.NORMAL,
                         dataStore=dataStore)
data_loader.debug_info()
validate_data = data_loader.validate_data
all_id_to_word = data_loader.idMapped.all_id_to_word
all_word_to_id = data_loader.idMapped.all_word_to_id
renameChain = data_loader.renameChain
rename_md_id_set = renameChain.rename_md_id_set
rename_data = renameChain.data
commit_hash_name_map = renameChain.commit_hash_name_map
validate_data_commit_hash_list = renameChain.validate_data_commit_hash_list
count_hit = 0
for i in range(len(validate_data)):
    commit_hash = validate_data_commit_hash_list[i]
    for ch in commit_hash_name_map:
        if ch == commit_hash:
            count_hit += 1
print('count_hit', count_hit)

# print('validate_data', validate_data)
print('rename_md_id_set', len(rename_md_id_set), rename_md_id_set)
count = 0
sub_count = 0
# for i in commit_hash_name_map:
#     # print(i, commit_hash_name_map[i])
#     for j in range(len(commit_hash_name_map[i])):
#         map_item = commit_hash_name_map[i][j]
#         if map_item['newest_name'] != map_item['new_name']:
#             print(map_item)
#             sub_count += 1
#     count += 1
# java/org/apache/catalina/tribes/transport/RxTaskPool#public_void_setMaxTasks(int_maxThreads)
# java/org/apache/catalina/tribes/transport/RxTaskPool#public_void_setMaxThreads(int_maxThreads)
print('count', count, sub_count)
# for i in range(len(rename_data)):
#     map_data = rename_data[i]['list']
#     print(i, map_data)
result_count = 0
for i in range(len(validate_data)):
    transaction = validate_data[i]
    for j in range(len(transaction)):
        item = transaction[j]
        final_name = all_id_to_word[item]
        cur_name = renameChain.get_cur_name(i, item)
        if final_name != cur_name:
            # print(i, final_name, cur_name)
            result_count += 1
print('result_count', result_count)


