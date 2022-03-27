from data.data_loader import get_vocab, get_module_data
# train, validate = get_contexts_target('LCExtractor')
# print('train contexts', train['contexts'])
# print('train target', train['target'])
# for i in range(len(train['contexts'])):
#     print(train['contexts'][i], train['target'][i])
# print('validate', validate)

# vocab = get_vocab('tomcat')
# print(len(vocab))

module_data_list = get_module_data('tomcat')
print(len(module_data_list), module_data_list[0])