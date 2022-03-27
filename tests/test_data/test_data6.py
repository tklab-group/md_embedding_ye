def get_method_map():
    method_map = [
        {
            'item': 'TestDao#query()',
            'index': 8
        },
        {
            'item': 'TestService#login()',
            'index': 7
        },
        {
            'item': 'TestController#login()',
            'index': 6
        },
        {
            'item': 'ClientDao#query()',
            'index': 5
        },
        {
            'item': 'ClientService#login()',
            'index': 4
        },
        {
            'item': 'ClientController#login()',
            'index': 3
        },
        {
            'item': 'UserDao#query()',
            'index': 2
        },
        {
            'item': 'UserService#login()',
            'index': 1
        },
        {
            'item': 'UserController#login()',
            'index': 0
        }
    ]
    return method_map


def get_module_data():
    md_list = [
        # validate data
        {
            'list': [
                6, 7, 8
            ],
        },
        # train data
        {
            'list': [
                3, 4, 5
            ],
        },
        {
            'list': [
                0, 1, 2
            ],
        },
        {
            'list': [
                0, 1, 2
            ],
        },
    ]
    return md_list
