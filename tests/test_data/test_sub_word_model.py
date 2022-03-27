import sys
sys.path.append('../../')
from tests.data.test_data4 import get_method_map, get_module_data
from model.main import Main
import torch
from data.mode_enum import Mode


if __name__ == '__main__':
    md_list = get_module_data()
    method_map = get_method_map()
    expected_validate_length = 3
    main = Main(dim=10, is_test=True, test_md_list=md_list, batch_size=2, max_epoch=10,
                test_method_map=method_map, expected_validate_length=expected_validate_length,
                mode=Mode.SUB_WORD)

    is_can_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_can_cuda else "cpu")
    main.train(is_load_model=False, load_path='', device=device, is_load_from_pkl=False)
    data_loader = main.data_loader
    # word_to_index = data_loader.vocab.word_to_index
    # sub_word_to_index = data_loader.vocab.sub_word_to_index
    # print(word_to_index)
    # print(sub_word_to_index)
    embeddingIndexMapped = data_loader.embeddingIndexMapped
    print(embeddingIndexMapped.word_to_in_embedding_index)

    dataDivider = data_loader.dataDivider
    train_data = dataDivider.get_train_data()
    validate_data = dataDivider.get_validate_data()
    print('train_data', train_data)
    print('validate_data', validate_data)
    train_contexts_target = data_loader.train_contexts_target
    # print('contexts', contexts)
    # print('target', target)
    print(train_contexts_target)
    main.eval(k=2)

