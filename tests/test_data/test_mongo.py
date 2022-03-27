import sys
sys.path.append('../../')
from data.mongo import MethodMapDao
from data.mongo import ModuleDataDao
from data.mongo import CoChangeDao

git_name = 'LCExtractor'
dao = ModuleDataDao()
doc_list = dao.query(git_name)
result = []
for doc in doc_list:
    # print(doc)
    module_list = doc['list']
    for i in range(len(module_list)):
        print(module_list[i])
        result.append(module_list[i])
print(len(result))
