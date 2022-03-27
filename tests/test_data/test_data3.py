def get_method_map():
    method_map = [
        {
            'item': 'com/config/DaoConfig#public_String_getConfig()',
            'index': 4
        },
        {
            'item': 'com/controller/UserController#public_Response_login(String_name,String_password)',
            'index': 3
        },
        {
            'item': 'com/dao/UserDao#public_UserPojo_query(String_name)',
            'index': 2
        },
        {
            'item': 'com/service/UserService#public_boolean_login(String_name,String_password)',
            'index': 1
        },
        {
            'item': 'com/pojo/UserPojo#public_String_getName()',
            'index': 0
        }
    ]
    return method_map


def get_module_data():
    md_list = [
        # validate data
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
