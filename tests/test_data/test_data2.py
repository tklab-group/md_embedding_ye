def get_method_map():
    method_map = [
        {
            'item': 'G',
            'index': 7
        },
        {
            'item': 'F',
            'index': 6
        },
        {
            'item': 'E',
            'index': 5
        },
        {
            'item': 'D',
            'index': 4
        },
        {
            'item': 'C',
            'index': 3
        },
        {
            'item': 'B',
            'index': 2
        },
        {
            'item': 'A',
            'index': 1
        },
        {
            'item': 'O',
            'index': 0
        }
    ]
    return method_map


def get_module_data():
    md_list = [
        # validate data
        {
            'list': [
                5, 7
            ],
        },
        {
            'list': [
                # O, E
                0, 5
            ],
        },
        {
            'list': [
                # A, C
                1, 3
            ],
        },
        # train data
        {
            'list': [
                0, 4, 5
            ],
        },
        {
            'list': [
                0, 5
            ],
        },
        {
            'list': [
                0, 4
            ],
        },
        {
            'list': [
                1, 2, 3
            ],
        },
        {
            'list': [
                2, 3
            ],
        },
        {
            'list': [
                1, 3
            ],
        },
        {
            'list': [
                1, 2
            ],
        }
    ]
    return md_list
