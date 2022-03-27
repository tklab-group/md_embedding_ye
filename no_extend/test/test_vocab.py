import sys
sys.path.append('../../../')
from data.data_loader import DataLoader

git_name = "tomcat"
validate_data_num = 1000
data_loader = DataLoader(git_name, validate_data_num, False)
# print(data_loader.vocab, min(data_loader.vocab.md_ids), max(data_loader.vocab.md_ids), data_loader.vocab.all_md_id_to_word[0])
data_loader.count_average_num()
vocab_size = data_loader.vocab.corpus_length
print('vocab_size', vocab_size)
print('all_size', len(data_loader.all_corpus))
print('cal_use_commit_count', data_loader.cal_use_commit_count())

