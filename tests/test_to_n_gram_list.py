import sys
sys.path.append('../')
from common.util import to_n_gram_list
import re
import datetime

word = 'org.apache.catalina.session.TestPersistentManagerIntegration#public_void_noSessionCreate_57637()'
result = to_n_gram_list(word, n=3)
print(result)

word2 = 'get'
result = to_n_gram_list(word2, n=3)
print(result)

word3 = 'http2section_5_2'
result = []
# subword = re.sub(r'[0-9]+', '', word3)
subword = word3
subword_split = subword.split('_')
for j in range(len(subword_split)):
    item = re.sub(r'[0-9]+', '', subword_split[j])
    if len(item) > 0:
        result.append(item)
print(result)
print(datetime.datetime.now())