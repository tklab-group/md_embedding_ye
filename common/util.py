import re
# import pickle
import pickle5 as pickle
import numpy as np
import sys
from collections import Counter
import time
import copy
import matplotlib.pyplot as plt
sys.path.append('../')


def save_data(data, pkl_file_path):
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(data, f, -1)


def load_data(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def leave_one_out(transaction):
    pair_list = []
    size = len(transaction)
    idx = np.arange(1, size) - np.tri(size, size - 1, k=-1, dtype=bool)
    query_list = np.array(transaction)[idx]
    length = len(query_list)
    for i in range(length):
        pair_list.append({
            'contexts': query_list[i][:],
            'target': transaction[i]
        })
    return pair_list


def to_sub_word(word):
    return set(to_sub_word_list(word))


def to_sub_word_list(word):
    trans = str.maketrans({
        '/': '_',
        '(': '_',
        ')': '_',
        ',': '_',
        '#': '_',
        '[': '_',
        ']': '_'
    })
    underline_word = hump2underline(word)
    underline_word = underline_word.translate(trans)
    result = underline_word.split('_')
    return result


def hump2underline(hunp_str):
    p = re.compile(r'([a-z]|\d)([A-Z])')
    sub = re.sub(p, r'\1_\2', hunp_str).lower()
    return sub


# https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python
def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def to_n_gram(word, n=4):
    return set(to_n_gram_list(word, n))


def to_n_gram_list(word, n=3, is_container_self=False):
    result = list()
    word_len = len(word)
    if word_len <= n:
        result.append('<' + word + '>')
        return result
    if is_container_self:
        result.append('<' + word + '>')
    result.append('<' + word[0: n - 1])
    for i in range(word_len):
        cur_word = word[i: i + n]
        if len(cur_word) == n:
            result.append(word[i: i + n])
    result.append(word[word_len - n + 1: word_len] + '>')
    return result


def n_gram_from_sub_word_set(sub_word_set, n=4):
    result = set()
    for item in iter(sub_word_set):
        n_gram_set = to_n_gram(item, n)
        # print(item, n_gram_set)
        result = result | n_gram_set
    return result


def n_gram_from_sub_word_list(sub_word_list, n=4):
    result = []
    for i in range(len(sub_word_list)):
        item = sub_word_list[i]
        n_gram_list = to_n_gram_list(item, n)
        result = result + n_gram_list
    return result


def get_common_prefix(str_a, str_b, stop_word=None):
    len_a = len(str_a)
    len_b = len(str_b)
    str_short = ''
    str_long = ''
    target_index = -1
    if len_a < len_b:
        str_short = str_a
        str_long = str_b
    else:
        str_short = str_b
        str_long = str_a
    for i in range(len(str_short)):
        if str_short[i].isupper():
            break
        if str_short[i] == str_long[i]:
            if (stop_word and str_short[i] == stop_word) or stop_word is None:
                target_index = i
        else:
            break
    return str_short[0: target_index + 1]


def get_common_prefix_for_list(str_list_a, str_list_b):
    len_a = len(str_list_a)
    len_b = len(str_list_b)
    str_list_short = []
    str_list_long = []
    target_index = -1
    if len_a < len_b:
        str_list_short = str_list_a
        str_list_long = str_list_b
    else:
        str_list_short = str_list_b
        str_list_long = str_list_a
    for i in range(len(str_list_short)):
        if str_list_short[i] == str_list_long[i]:
            target_index = i
        else:
            break
    return str_list_short[0: target_index + 1]


def get_package_sub_word_list(package_list):
    package_sub_word_list = []
    for i in range(len(package_list)):
        package = package_list[i]
        package_split = package.split('/')
        package_sub_word_list.append(package_split)
    return package_sub_word_list


def get_freq_package_common_part(package_list, threshold=0.1):
    start_time = time.time()
    train_words = len(package_list)
    package_sub_word_list = get_package_sub_word_list(package_list)

    is_break = False
    while_count = 0
    score_counter = {}
    while True:
        if is_break:
            break
        bi_gram_counter = {}
        # counter
        for i in range(len(package_sub_word_list)):
            item = package_sub_word_list[i]
            # print('item', item)
            # for j in range(len(item)):
            for j in range(1):
                if j + 1 < len(item):
                    bi_gram = item[j] + '/' + item[j + 1]
                    # print('bi_gram', bi_gram)
                    if bi_gram in bi_gram_counter:
                        bi_gram_counter[bi_gram] += 1
                    else:
                        bi_gram_counter[bi_gram] = 1
        # print('train_words', train_words)
        # scoreを計算
        for bi_gram in iter(bi_gram_counter.keys()):
            # print('bi_gram', bi_gram)
            bi_gram_count = bi_gram_counter[bi_gram]
            score = bi_gram_count / train_words
            if score >= threshold:
                # print(bi_gram, score)
                score_counter[bi_gram] = score
        # merge
        is_have_change = False
        for i in range(len(package_sub_word_list)):
            item = package_sub_word_list[i]
            new_list = []
            is_skip = False
            if len(item) > 1:
                for j in range(len(item)):
                    if is_skip:
                        is_skip = False
                        continue
                    if j + 1 < len(item):
                        bi_gram = item[j] + '/' + item[j + 1]
                        # print('bi_gram', bi_gram)
                        if bi_gram in score_counter:
                            new_list.append(bi_gram)
                            is_skip = True
                            is_have_change = True
                    if not is_skip:
                        new_list.append(item[j])
                # print('new_list', new_list, item)
                new_list_str = '/'.join(new_list)
                item_str = '/'.join(item)
                if new_list_str != item_str:
                    print(new_list_str, item_str)
                package_sub_word_list[i] = new_list
        is_break = not is_have_change
        # print(sort_counter(bi_gram_counter))
        # print(bi_gram_counter)
        while_count += 1
    # print('while_count', while_count)
    # sorted_score_counter = sort_counter(score_counter)
    sorted_score_counter = sorted(score_counter.items(), key=lambda kv: (kv[0]), reverse=True)
    result = {}
    for i in range(len(sorted_score_counter)):
        item = sorted_score_counter[i]
        result[item[0]] = item[1]
    end_time = time.time() - start_time
    # print('cost time', end_time)
    return result


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


def sort_counter(counter):
    return sorted(counter.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)


def decode_module_data(module_data):
    module_data_split = module_data.split('#', 1)

    package_class = module_data_split[0]
    package_class_split = package_class.split('.')

    if len(package_class_split) > 1:
        package_sub_word = package_class_split[0:len(package_class_split) - 1]
        package = '/'.join(package_sub_word)

        class_name = package_class_split[len(package_class_split) - 1]

    else:
        package = ''

        if len(package_class_split) == 1:
            class_name = package_class_split[0]
        else:
            class_name = ''
    method_signature = module_data_split[1]
    return package, clean_class_name(class_name), method_signature


def delete_method_modifiers(method_signature):
    if not method_signature:
        return ''
    modifiers = ['synchronized_', 'transient_', 'protected_',
                 'abstract_', 'volatile_', 'private_',
                 'static_', 'public_', 'final_']
    for i in range(len(modifiers)):
        method_signature = method_signature.replace(modifiers[i], '')
    return method_signature


def clean_return_type(return_type,
                      is_delete_void_return_type=True,
                      is_number_type_token_return_type=True):
    # return_type = return_type.split('[')[0]
    if len(return_type) == 1:
        return_type = ''
    if is_delete_void_return_type and return_type == 'void':
        return_type = ''
    if is_number_type_token_return_type:
        number_class = ['int', 'long', 'byte', 'float', 'double', 'short',
                        'Number', 'Integer', 'Long', 'Float', 'Double', 'Byte', 'Short'
                        ]
        for i in range(len(number_class)):
            if number_class[i] == return_type:
                return_type = '<NUMBERTYPE>'
                break
    return_type = return_type.replace('.', '')
    return return_type


def clean_method_name(method_name,
                      is_constructor=False,
                      is_delete_number_method_name=True):
    if is_constructor:
        return method_name
    if is_delete_number_method_name:
        # delete all number
        method_name = re.sub(r'[0-9]+', '', method_name)
    # delete prefix 'test'
    return method_name


# method_signature after delete_method_modifiers
def check_is_constructor(method_signature):
    result = True
    for i in range(len(method_signature)):
        # ClassName()
        if method_signature[i] == '(':
            break
        # ReturnType_methodName...
        if method_signature[i] == '_':
            result = False
            break
    return result


# Type typeName,Type2, typeName2,...
def split_param(param):
    if not param:
        return []
    return param.split(',')


def clean_class_name(class_name):
    return class_name.replace('[', '').replace(']', '')


def clean_type_name(type_name,
                    is_delete_number_param_name=True,
                    is_delete_single_token_param_name=True):
    # type_name = type_name.split('[', 1)[0]
    type_name = type_name.split('.')[0]
    if is_delete_number_param_name:
        type_name = re.sub(r'[0-9]+', '', type_name)
    if is_delete_single_token_param_name and len(type_name) == 1:
        return ''
    return type_name


def clean_param(param,
                is_number_type_token_param_type=True,
                is_delete_number_param_name=True,
                is_delete_single_token_param_name=True):
    result = []
    split_param_list = split_param(param)
    for i in range(len(split_param_list)):
        item = split_param_list[i]
        item_split = item.split('_', 1)
        cleaned_type = clean_return_type(
            return_type=item_split[0],
            is_number_type_token_return_type=is_number_type_token_param_type
        )
        cleaned_type_name = clean_type_name(
            type_name=item_split[1],
            is_delete_number_param_name=is_delete_number_param_name,
            is_delete_single_token_param_name=is_delete_single_token_param_name
        )
        result.append([cleaned_type, cleaned_type_name])
    return result


def decode_method_signature(
        method_signature,
        # modifier
        is_delete_modifier=True,
        # return type
        is_delete_void_return_type=True,
        is_number_type_token_return_type=False,
        # method name
        is_delete_number_method_name=False,
        # param type
        is_number_type_token_param_type=False,
        # param name
        is_delete_number_param_name=False,
        is_delete_single_token_param_name=True,
):
    # print('method_signature', method_signature)
    if is_delete_modifier:
        method_signature = delete_method_modifiers(method_signature)
    # delete all [...] content
    stack = []
    delete_word_set = set()
    start_delete_index = -1
    for i in range(len(method_signature)):
        if method_signature[i] == '[':
            if len(stack) == 0:
                start_delete_index = i
                # print('start', method_signature[0: i + 1])
            stack.append('[')
        if method_signature[i] == ']':
            stack.pop()
            if len(stack) == 0:
                delete_word_set.add(method_signature[start_delete_index: i + 1])
                # print('check', method_signature[start_delete_index: i + 1])
    delete_word_list = list(delete_word_set)
    # print('delete_word_list', delete_word_list)
    delete_word_list.sort(key=lambda word: len(word), reverse=True)
    for i in range(len(delete_word_list)):
        method_signature = method_signature.replace(delete_word_list[i], '')
    # print('sort delete_word_list', delete_word_list)
    method_signature = method_signature.replace('__', '_')
    if method_signature.startswith('_'):
        method_signature = method_signature[1: len(method_signature)]

    is_constructor = check_is_constructor(method_signature)
    if is_constructor:
        # print('constructor', method_signature)
        return_type = ''

        method_name_other_split = method_signature.split('(', 1)
        method_name = method_name_other_split[0]
    else:

        return_type_other_split = method_signature.split('_', 1)
        return_type = return_type_other_split[0]
        # print('return_type', return_type)

        method_name_other_split = return_type_other_split[1].split('(', 1)
        method_name = method_name_other_split[0]
    # print(method_signature, method_name_other_split)
    # print()
    param = method_name_other_split[1].rstrip(')')
    return clean_return_type(
        return_type=return_type,
        is_number_type_token_return_type=is_number_type_token_return_type,
        is_delete_void_return_type=is_delete_void_return_type
    ), clean_method_name(
        method_name=method_name,
        is_constructor=is_constructor,
        is_delete_number_method_name=is_delete_number_method_name
    ), clean_param(
        param=param,
        is_delete_single_token_param_name=is_delete_single_token_param_name,
        is_delete_number_param_name=is_delete_number_param_name,
        is_number_type_token_param_type=is_number_type_token_param_type
    )


# 元々のFileの定義はpackage.class_name、しかし、ここではFileをModule dataの抽象化したタイプでも見なす
def get_file_level_info(module_data):
    module_data_split = module_data.split('#', 1)
    package_class = module_data_split[0]
    return package_class

    # package, class_name, method_signature = decode_module_data(module_data)
    # return_type, method_name, split_param_list = decode_method_signature(method_signature)
    # return package + '.' + class_name + '#' + method_name

    # package, class_name, method_signature = decode_module_data(module_data)
    # return_type, method_name, split_param_list = decode_method_signature(method_signature)
    # return method_name


def save_boxplot(git_name,
                 model_name,
                 recall_type,
                 recall_type_path,
                 tarmaq_recall,
                 cbow_recall_list):
    # 箱引け図
    fig, ax = plt.subplots()
    title = git_name + ' ' + recall_type
    ax.set_title(title)
    label_1 = 'TARMAQ'
    label_2 = model_name
    ax.set_xticklabels([label_1, label_2])
    ax.boxplot(([tarmaq_recall], cbow_recall_list),
               showmeans=True, widths=0.6)
    # ax.set_aspect(1.5)
    fname = git_name + '_' + recall_type_path
    plt.savefig('./fig/' + fname)
