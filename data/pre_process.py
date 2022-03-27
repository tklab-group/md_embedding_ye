import sys

sys.path.append('../')
from collections import Counter
from data.mode_enum import Mode
from data.rename_chain import RenameChain
from common.util import to_sub_word_list, n_gram_from_sub_word_list, \
    remove_prefix, sort_counter, decode_module_data, decode_method_signature, \
    get_freq_package_common_part, get_common_prefix, hump2underline, get_file_level_info, camel_case_split
import re


# only handle sub word method
class PreProcess:
    def __init__(self,
                 train_data,
                 train_id_to_word,
                 train_words,
                 train_data_commit_hash_list,
                 renameChain: RenameChain,
                 mode=Mode.NORMAL,
                 is_use_package=True,
                 is_use_class_name=True,
                 is_use_return_type=True,
                 is_use_method_name=True,
                 is_use_param_type=True,
                 is_use_param_name=True,
                 is_check_rename=True,
                 is_simple_handle_package=False,
                 is_simple_handle_class_name=False,
                 is_simple_handle_return_type=False,
                 is_simple_handle_method_name=False,
                 is_simple_handle_param_type=False,
                 is_simple_handle_param_name=False,
                 is_predict_with_file_level=False,
                 is_mark_respective_type=False,
                 # check preprocessing
                 is_preprocessing_package=True,
                 is_delete_modifier=True,
                 is_delete_void_return_type=True,
                 is_casing=True,
                 is_delete_single_subword=True,
                 is_delete_number_from_method_and_param=False,
                 is_number_type_token_from_return_and_param_type=False,
                 is_delete_sub_word_number=True,
                 is_clean_sub_word=True,
                 is_print_info=False,
                 ):
        # simply
        is_delete_number_method_name = is_delete_number_from_method_and_param
        is_delete_number_param_name = is_delete_number_from_method_and_param
        is_delete_single_token_param_name = is_delete_single_subword
        is_number_type_token_param_type = is_number_type_token_from_return_and_param_type
        is_number_type_token_return_type = is_number_type_token_from_return_and_param_type
        if is_print_info and is_preprocessing_package and is_delete_modifier and is_delete_void_return_type and is_casing and \
                is_delete_single_subword and not is_delete_number_from_method_and_param and \
                not is_number_type_token_from_return_and_param_type and is_delete_sub_word_number:
            print('default preprocess config')
        if is_print_info:
            print('pre_process info start', '-'*16)
            print(is_use_package, is_use_class_name, is_use_return_type, is_use_method_name, is_use_param_type,
                  is_use_param_name)
            if is_preprocessing_package:
                print('is_preprocessing_package')
            if is_delete_modifier:
                print('is_delete_modifier')
            if is_delete_void_return_type:
                print('is_delete_void_return_type')
            if is_casing:
                print('is_casing')
            if is_delete_single_subword:
                print('is_delete_single_subword')
            if is_delete_number_from_method_and_param:
                print('is_delete_number_from_method_and_param')
            if is_number_type_token_from_return_and_param_type:
                print('is_number_type_token_from_return_and_param_type')
            print('is_clean_sub_word', is_clean_sub_word)
            print('is_delete_sub_word_number', is_delete_sub_word_number)
            print('is_mark_respective_type', is_mark_respective_type)
            print('pre_process info end', '-' * 16)
        self.train_data = train_data
        self.train_id_to_word = train_id_to_word
        self.train_words = train_words
        self.train_data_commit_hash_list = train_data_commit_hash_list
        self.renameChain = renameChain
        self.mode = mode
        self.is_use_package = is_use_package
        self.is_use_class_name = is_use_class_name
        self.is_use_return_type = is_use_return_type
        self.is_use_method_name = is_use_method_name
        self.is_use_param_type = is_use_param_type
        self.is_use_param_name = is_use_param_name
        self.is_check_rename = is_check_rename
        self.is_simple_handle_package = is_simple_handle_package
        self.is_simple_handle_class_name = is_simple_handle_class_name
        self.is_simple_handle_return_type = is_simple_handle_return_type
        self.is_simple_handle_method_name = is_simple_handle_method_name
        self.is_simple_handle_param_type = is_simple_handle_param_type
        self.is_simple_handle_param_name = is_simple_handle_param_name
        self.is_predict_with_file_level = is_predict_with_file_level
        self.is_mark_respective_type = is_mark_respective_type
        self.is_delete_sub_word_number = is_delete_sub_word_number
        self.is_clean_sub_word = is_clean_sub_word

        self.is_preprocessing_package = is_preprocessing_package
        self.is_delete_modifier = is_delete_modifier
        self.is_delete_void_return_type = is_delete_void_return_type
        self.is_number_type_token_return_type = is_number_type_token_return_type
        self.is_delete_number_method_name = is_delete_number_method_name
        self.is_number_type_token_param_type = is_number_type_token_param_type
        self.is_delete_number_param_name = is_delete_number_param_name
        self.is_delete_single_token_param_name = is_delete_single_token_param_name
        self.is_casing = is_casing
        self.is_delete_single_subword = is_delete_single_subword

        # common prefix part from train data
        self.common_prefix = ''

        self.package_class_list = []
        self.get_package_class_list()

        if mode == Mode.NORMAL:
            return
        self._find_common_prefix_from_train_data()

        self.package_token = '<package>'
        self.class_name_token = '<class>'
        self.return_type_token = '<return_type>'
        self.method_name_token = '<method>'
        self.param_type_token = '<param_type>'
        self.param_name_token = '<param_name>'

        self.package_list = []
        self.class_name_list = []
        self.method_signature_list = []

        self.return_type_list = []
        self.method_name_list = []
        self.param_type_list = []
        self.param_name_list = []

        self._decode_module_data()

        if self.is_preprocessing_package:
            self.freq_package_common_part = get_freq_package_common_part(
                package_list=self.package_list,
                threshold=0.1
            )
            # {'org/apache/tomcat/util': 0.17512057209379678, 'org/apache/tomcat': 0.2843838350241144, 'org/apache/coyote': 0.1865624480292699, 'org/apache/catalina': 0.39856976550806583, 'org/apache': 0.9489439547646765}
            # print(self.freq_package_common_part)
            # test1 = 'org.apache.tomcat.util'
            # count1 = 0
            # test2 = 'org.apache.catalina'
            # count2 = 0
            # for word in self.train_words:
            #     if count1 < 4 and word.find(test1) != -1:
            #         print(word)
            #         count1 += 1
            #     if count2 < 4 and word.find(test2) != -1:
            #         print(word)
            #         count2 += 1
        else:
            self.freq_package_common_part = {}
        # self.debug()

    def _find_common_prefix_from_train_data(self):
        common_prefix = ''
        index = 0
        for word in iter(self.train_words):
            package, class_name, method_signature = decode_module_data(word)
            if index == 0:
                common_prefix = package
            else:
                common_prefix = get_common_prefix(common_prefix, package, '/')
            index += 1
            # print('package', package)
            # print('word', word)
            # print('common_prefix', common_prefix)
        self.common_prefix = common_prefix

    def get_package_class_list(self):
        package_class_list = []
        if len(self.package_class_list) == 0 and self.is_predict_with_file_level:
            for i in range(len(self.train_data)):
                id_list = self.train_data[i]
                for j in range(len(id_list)):
                    md_id = id_list[j]
                    if self.is_check_rename:
                        word = self.renameChain.get_cur_name_by_hash(self.train_data_commit_hash_list[i], md_id)
                    else:
                        word = self.train_id_to_word[md_id]
                    package_class_list.append(get_file_level_info(word))
            self.package_class_list = package_class_list

    def _decode_module_data(self):
        package_list = []
        class_name_list = []
        method_signature_list = []

        return_type_list = []
        method_name_list = []
        param_type_list = []
        param_name_list = []
        for i in range(len(self.train_data)):
            id_list = self.train_data[i]
            for j in range(len(id_list)):
                md_id = id_list[j]
                if self.is_check_rename:
                    word = self.renameChain.get_cur_name_by_hash(self.train_data_commit_hash_list[i], md_id)
                else:
                    word = self.train_id_to_word[md_id]
                # common_prefixを削除
                if self.common_prefix:
                    word = remove_prefix(word, self.common_prefix)
                # dirty_data = ['service]',
                #               'proc[fake',
                #               '[']
                # for d in range(len(dirty_data)):
                #     if word.lower().find(dirty_data[d]) != -1:
                #         # print('dirty_data', dirty_data[d], word)
                #         print('\''+word+'\',')
                package, class_name, method_signature = decode_module_data(word)
                return_type, method_name, split_param_list = decode_method_signature(
                    method_signature=method_signature,
                    # modifier
                    is_delete_modifier=self.is_delete_modifier,
                    # return type
                    is_delete_void_return_type=self.is_delete_void_return_type,
                    is_number_type_token_return_type=self.is_number_type_token_return_type,
                    # method name
                    is_delete_number_method_name=self.is_delete_number_method_name,
                    # param type
                    is_number_type_token_param_type=self.is_number_type_token_param_type,
                    # param name
                    is_delete_number_param_name=self.is_delete_number_param_name,
                    is_delete_single_token_param_name=self.is_delete_single_token_param_name,
                )

                if self.is_use_package:
                    package_list.append(package)
                if self.is_use_class_name:
                    class_name_list.append(class_name)
                method_signature_list.append(method_signature)
                if self.is_use_return_type:
                    return_type_list.append(return_type)
                if self.is_use_method_name:
                    method_name_list.append(method_name)

                for k in range(len(split_param_list)):
                    item = split_param_list[k]
                    if self.is_use_param_type:
                        param_type_list.append(item[0])
                    if self.is_use_param_name:
                        param_name_list.append(item[1])
        self.package_list = package_list
        self.class_name_list = class_name_list
        self.method_signature_list = method_signature_list

        self.return_type_list = return_type_list
        self.method_name_list = method_name_list
        self.param_type_list = param_type_list
        self.param_name_list = param_name_list

    def get_package_sub_word(self, package):
        if not package:
            return []
        if not self.is_use_package:
            return []
        if self.is_simple_handle_package:
            if self.is_mark_respective_type:
                return [self.package_token + package]
            else:
                return [package]
        result = []
        if self.is_preprocessing_package:
            is_hit = False
            for package_part in iter(self.freq_package_common_part):
                if package.startswith(package_part):
                    if self.is_mark_respective_type:
                        result.append(self.package_token + package_part)
                    else:
                        result.append(package_part)
                    other_package_part = package.split(package_part, 1)[1]
                    other_package_part_split = other_package_part.split('/')
                    for i in range(len(other_package_part_split)):
                        if other_package_part_split[i]:
                            if self.is_mark_respective_type:
                                result.append(self.package_token + other_package_part_split[i])
                            else:
                                result.append(other_package_part_split[i])
                    is_hit = True
                    break
            if not is_hit:
                package_split = package.split('/')
                for i in range(len(package_split)):
                    if self.is_mark_respective_type:
                        result.append(self.package_token + package_split[i])
                    else:
                        result.append(package_split[i])
        else:
            package_split = package.split('/')
            for i in range(len(package_split)):
                if self.is_mark_respective_type:
                    result.append(self.package_token + package_split[i])
                else:
                    result.append(package_split[i])
        return result

    def get_sub_word(self, word):
        if not word:
            return []
        # result = []
        # if self.is_casing:
        #     result = camel_case_split(word)
        # else:
        #     temp = hump2underline(word)
        #     result = temp.split('_')
        result = camel_case_split(word)
        if self.is_clean_sub_word:
            real_result = []
            for i in range(len(result)):
                subword = result[i]
                subword_split = subword.split('_')
                for j in range(len(subword_split)):
                    item = subword_split[j]
                    # delete all number
                    # item = re.sub(r'[0-9]+', '', item)
                    if len(item) > 0 and not (self.is_delete_sub_word_number and item.isdigit()):
                        real_result.append(item)
            result = real_result
        if not self.is_casing:
            for i in range(len(result)):
                result[i] = result[i].lower()
        if self.is_delete_single_subword:
            final_result = []
            for i in range(len(result)):
                if len(result[i]) > 1:
                    final_result.append(result[i])
            return final_result
        else:
            return result

    def get_class_name_sub_word(self, class_name):
        if not class_name:
            return []
        if not self.is_use_class_name:
            return []
        if self.is_simple_handle_class_name:
            if self.is_mark_respective_type:
                return [self.class_name_token + class_name]
            else:
                return [class_name]
        sub_word_list = self.get_sub_word(class_name)
        if self.is_mark_respective_type:
            for i in range(len(sub_word_list)):
                sub_word_list[i] = self.class_name_token + sub_word_list[i]
            return sub_word_list
        else:
            return sub_word_list

    def get_return_type_sub_word(self, return_type):
        if not return_type:
            return []
        if not self.is_use_return_type:
            return []
        if self.is_simple_handle_return_type:
            if self.is_mark_respective_type:
                return [self.return_type_token + return_type]
            else:
                return [return_type]
        if return_type == '<NUMBERTYPE>':
            if self.is_mark_respective_type:
                return [self.return_type_token + return_type]
            else:
                return [return_type]
        sub_word_list = self.get_sub_word(return_type)
        if self.is_mark_respective_type:
            for i in range(len(sub_word_list)):
                sub_word_list[i] = self.return_type_token + sub_word_list[i]
            return sub_word_list
        else:
            return sub_word_list

    def get_method_name_sub_word(self, method_name):
        sub_word_list = self.get_sub_word(method_name)
        if self.is_mark_respective_type:
            for i in range(len(sub_word_list)):
                sub_word_list[i] = self.method_name_token + sub_word_list[i]
            return sub_word_list
        else:
            return sub_word_list

    def get_param_type_sub_word(self, param_type):
        if not param_type:
            return []
        if not self.is_use_param_type:
            return []
        if self.is_simple_handle_param_type:
            if self.is_mark_respective_type:
                return [self.param_type_token + param_type]
            else:
                return [param_type]
        if param_type == '<NUMBERTYPE>':
            if self.is_mark_respective_type:
                return [self.param_type_token + param_type]
            else:
                return [param_type]
        sub_word_list = self.get_sub_word(param_type)
        if self.is_mark_respective_type:
            for i in range(len(sub_word_list)):
                sub_word_list[i] = self.param_type_token + sub_word_list[i]
            return sub_word_list
        else:
            return sub_word_list

    def get_param_name_sub_word(self, param_name):
        if not param_name:
            return []
        if not self.is_use_param_name:
            return []
        if self.is_simple_handle_param_name:
            if self.is_mark_respective_type:
                return [self.param_name_token + param_name]
            else:
                return [param_name]
        sub_word_list = self.get_sub_word(param_name)
        if self.is_mark_respective_type:
            for i in range(len(sub_word_list)):
                sub_word_list[i] = self.param_name_token + sub_word_list[i]
            return sub_word_list
        else:
            return sub_word_list

    def get_module_data_sub_word(self, module_data):
        if self.common_prefix:
            module_data = remove_prefix(module_data, self.common_prefix)
        package, class_name, method_signature = decode_module_data(module_data)
        package_sub_word = self.get_package_sub_word(package)
        class_name_sub_word = self.get_class_name_sub_word(class_name)

        return_type, method_name, split_param_list = decode_method_signature(method_signature)
        return_type_sub_word = self.get_return_type_sub_word(return_type)
        method_name_sub_word = self.get_method_name_sub_word(method_name)
        param_type_sub_word = []
        param_name_sub_word = []
        for k in range(len(split_param_list)):
            item = split_param_list[k]

            temp_param_type_sub_word = self.get_param_type_sub_word(item[0])
            for j in range(len(temp_param_type_sub_word)):
                param_type_sub_word.append(temp_param_type_sub_word[j])

            temp_param_name_sub_word = self.get_param_name_sub_word(item[1])
            for j in range(len(temp_param_name_sub_word)):
                param_name_sub_word.append(temp_param_name_sub_word[j])

        result = []
        for i in range(len(package_sub_word)):
            result.append(package_sub_word[i])

        for i in range(len(class_name_sub_word)):
            result.append(class_name_sub_word[i])

        for i in range(len(return_type_sub_word)):
            result.append(return_type_sub_word[i])

        for i in range(len(method_name_sub_word)):
            result.append(method_name_sub_word[i])

        for i in range(len(param_type_sub_word)):
            result.append(param_type_sub_word[i])

        for i in range(len(param_name_sub_word)):
            result.append(param_name_sub_word[i])

        return result

    def debug(self):
        print('debug pre_process')
        # print('debug_package_class')
        # self.debug_package_class()
        # print('debug_freq_package_common_part')
        # self.debug_freq_package_common_part()
        # print('debug_package')
        # self.debug_package()
        print('debug_class_name')
        self.debug_class_name()
        # print('debug_return_type')
        # self.debug_return_type()
        # print('debug_method_signature')
        # self.debug_method_signature()
        # print('debug_method_name')
        # self.debug_method_name()
        # self.debug_param()
        # print('debug_param_type')
        # self.debug_param_type()
        # print('debug_param_name')
        # self.debug_param_name()

    def debug_freq_package_common_part(self):
        for word in iter(self.freq_package_common_part):
            print(word, self.freq_package_common_part[word])

    def debug_package(self):
        package_counter = Counter(self.package_list)
        sorted_list = package_counter.most_common()
        error_list = []
        single_list = []
        other_part_list = []
        package_sub_word_list = []
        for i in range(len(sorted_list)):
            item = sorted_list[i]
            package_sub_word = self.get_package_sub_word(item[0])
            print('package', item[0], package_sub_word, item[1])

            package_check = ''
            for j in range(len(package_sub_word)):
                package_check = package_check + package_sub_word[j]
            if package_check != item[0]:
                error_list.append(item[0])
            for q in range(len(package_sub_word)):
                package_sub_word_list.append(package_sub_word[q])

        package_sub_word_counter = Counter(package_sub_word_list)
        sort_package_sub_word_list = package_sub_word_counter.most_common()
        # print('error', error_list)

        for i in range(len(sort_package_sub_word_list)):
            item = sort_package_sub_word_list[i]
            print('package sub word', item[0], item[1])

        # for i in range(len(sort_other_part_list)):
        #     item = sort_other_part_list[i]
        #     print('other package', item[0], item[1])

    def debug_class_name(self):
        class_name_counter = Counter(self.class_name_list)
        sorted_list = class_name_counter.most_common()
        for i in range(len(sorted_list)):
            item = sorted_list[i]
            print('class name', item[0], item[1])

        class_name_sub_word_list = []
        for i in range(len(self.class_name_list)):
            item = self.class_name_list[i]
            sub_word_list = to_sub_word_list(item)
            for j in range(len(sub_word_list)):
                class_name_sub_word_list.append(sub_word_list[j])
        class_name_sub_word_counter = Counter(class_name_sub_word_list)
        sub_word_sorted_list = class_name_sub_word_counter.most_common()
        for i in range(len(sub_word_sorted_list)):
            item = sub_word_sorted_list[i]
            print('sub class name', item[0], item[1])

    def debug_method_signature(self):
        method_signature_counter = Counter(self.method_signature_list)
        sorted_list = method_signature_counter.most_common()
        for i in range(len(sorted_list)):
            item = sorted_list[i]
            print('method signature', item[0], item[1])

        # method_signature_sub_word_list = []
        # for i in range(len(method_signature_list)):
        #     item = method_signature_list[i]
        #     sub_word_list = to_sub_word_list(item)
        #     for j in range(len(sub_word_list)):
        #         method_signature_sub_word_list.append(sub_word_list[j])
        # method_signature_sub_word_counter = Counter(method_signature_sub_word_list)
        # sub_word_sorted_list = method_signature_sub_word_counter.most_common()
        # for i in range(len(sub_word_sorted_list)):
        #     item = sub_word_sorted_list[i]
        #     print('sub method signature', item[0], item[1])

    def debug_return_type(self):
        return_type_counter = Counter(self.return_type_list)
        sort_return_type_list = return_type_counter.most_common()
        for i in range(len(sort_return_type_list)):
            item = sort_return_type_list[i]
            print('return type', item[0], item[1])

    def debug_method_name(self):
        method_name_counter = Counter(self.method_name_list)
        sort_method_name_list = method_name_counter.most_common()
        for i in range(len(sort_method_name_list)):
            item = sort_method_name_list[i]
            if i < 2000:
                print('method name', item[0], item[1])

    def debug_param_type(self):
        type_counter = Counter(self.param_type_list)
        sort_type_list = type_counter.most_common()
        for i in range(len(sort_type_list)):
            item = sort_type_list[i]
            print('type', item[0], item[1])

    def debug_param_name(self):
        type_name_counter = Counter(self.param_name_list)
        sort_type_name_list = type_name_counter.most_common()
        for i in range(len(sort_type_name_list)):
            item = sort_type_name_list[i]
            print('type name', item[0], item[1])

    def debug_param(self):
        param_list = []
        for i in range(len(self.method_signature_list)):
            item = method_signature_list[i]
            return_type, method_name, split_param_list = decode_method_signature(item)
            for j in range(len(split_param_list)):
                param_list.append('_'.join(split_param_list[j]))
        param_counter = Counter(param_list)
        sort_param_list = param_counter.most_common()
        for i in range(len(sort_param_list)):
            item = sort_param_list[i]
            print('param', item[0], item[1])

    def debug_package_class(self):
        print('debug_package_class')
        package_class_counter = Counter(self.package_class_list)
        sort_package_class_list = package_class_counter.most_common()
        print('file size', len(sort_package_class_list))
        for i in range(len(sort_package_class_list)):
            item = sort_package_class_list[i]
            print('file', item[0], item[1])
