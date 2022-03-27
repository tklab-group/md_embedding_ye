import sys
sys.path.append('../')
import numpy as np


class RenameChain:
    def __init__(self,
                 validate_data,
                 validate_data_commit_hash_list,
                 id_to_word,
                 word_to_id,
                 data):
        self.validate_data = validate_data
        self.validate_data_commit_hash_list = validate_data_commit_hash_list
        self.id_to_word = id_to_word
        self.word_to_id = word_to_id
        self.data = data
        # print('data', data)

        # commit_hash => [ {newestName=>newName}, ... ]
        self.commit_hash_name_map = {}
        self.rename_md_id_set = set()
        self.build_map()
        # self.debug()

    def build_map(self):
        for i in range(len(self.data)):
            map_data = self.data[i]['list']
            # print(map_data)
            newest_name = map_data[len(map_data)-1]['newName']
            if newest_name in self.word_to_id:
                self.rename_md_id_set.add(self.word_to_id[newest_name])
            for j in range(len(map_data)):
                item = map_data[j]
                commit_hash = item['commitHash']
                old_name = item['oldName']
                new_name = item['newName']
                if commit_hash not in self.commit_hash_name_map:
                    self.commit_hash_name_map[commit_hash] = []
                self.commit_hash_name_map[commit_hash].append({
                    'newest_name': newest_name,
                    'new_name': new_name
                })

    def get_cur_name_by_hash(self, commit_hash, md_id):
        newest_name = self.id_to_word[md_id]
        if md_id not in self.rename_md_id_set:
            return newest_name
        if commit_hash in self.commit_hash_name_map:
            chain = self.commit_hash_name_map[commit_hash]
            for i in range(len(chain)):
                if newest_name == chain[i]['newest_name']:
                    return chain[i]['new_name']
        return newest_name

    def debug(self):
        for commit_hash in self.commit_hash_name_map:
            print(commit_hash, self.commit_hash_name_map[commit_hash])



