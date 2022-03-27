import sys
sys.path.append('../../')
from tests.data.test_data2 import get_method_map, get_module_data
from model.main import Main
import torch


if __name__ == '__main__':
    md_list = get_module_data()
    method_map = get_method_map()
    expected_validate_length = 3
    main = Main(dim=10, is_test=True, test_md_list=md_list, batch_size=2, max_epoch=10,
                test_method_map=method_map, expected_validate_length=expected_validate_length)

    is_can_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_can_cuda else "cpu")
    main.train(is_load_model=False, load_path='', device=device, is_load_from_pkl=False)

    main.eval(k=2)

