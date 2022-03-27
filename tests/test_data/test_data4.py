def get_method_map():
    method_map = [
        {
            'item': 'UserTestDao#query()',
            'index': 5
        },
        {
            'item': 'DaoConfig#getConfig()',
            'index': 4
        },
        {
            'item': 'UserController#login()',
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
            'item': 'UserPojo#getName()',
            'index': 0
        }
    ]
    return method_map


def get_module_data():
    md_list = [
        # validate data
        {
            'list': [
                # DaoConfig#getConfig() => 5, 10 16 14
                # UserPojo#getName() => 4, 7 13 14 15
                0, 4
            ],
        },
        {
            'list': [
                # UserController#login() => 3, 7 12 9
                # UserService#login() => 1, 7 8 9
                1, 3
            ],
        },
        {
            'list': [
                # DaoConfig#getConfig() => 5, 10 16 14
                # UserService#login() => 1, 7 8 9
                # UserPojo#getName() => 4, 7 13 14 15
                0, 4, 1
            ],
        },
        # train data
        {
            'list': [
                # UserPojo#getName() => 4, 7 13 14 15
                # UserService#login() => 1, 7 8 9
                0, 1
            ],
        },
        {
            'list': [
                # DaoConfig#getConfig() => 5, 10 16 14
                # UserPojo#getName() => 4, 7 13 14 15
                0, 4
            ],
        },
        {
            'list': [
                # UserController#login() => 3, 7 12 9
                # UserDao#query() => 2, 7 10 11
                # UserService#login() => 1, 7 8 9
                1, 2, 3
            ],
        },
        {
            'list': [
                # UserController#login() => 3, 7 12 9
                # UserDao#query() => 2, 7 10 11
                2, 3
            ],
        },
        {
            'list': [
                # UserController#login() => 3, 7 12 9
                # UserService#login() => 1, 7 8 9
                1, 3
            ],
        },
        {
            'list': [
                # UserDao#query() => 2, 7 10 11
                # UserService#login() => 1, 7 8 9
                1, 2
            ],
        }
    ]
    return md_list
