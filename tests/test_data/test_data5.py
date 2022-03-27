def get_method_map():
    method_map = [
        {
            'item': 'com/test/TestService#public_boolean_testLogin(String_name,String_password)',
            'index': 5
        },
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
    # 0 => pojo
    # 1 => service
    # 2 => dao
    # 3 => controller
    # 4 => config
    # 5 => test
    md_list = [
        # validate data
        {
            'list': [
                # pojo, test
                0, 5
            ],
        },
        {
            'list': [
                # service, controller
                1, 3
            ],
        },
        # train data
        {
            'list': [
                # pojo, config, test
                0, 4, 5
            ],
        },
        {
            'list': [
                # pojo, test
                0, 5
            ],
        },
        {
            'list': [
                # pojo, config
                0, 4
            ],
        },
        {
            'list': [
                # service, dao, controller
                1, 2, 3
            ],
        },
        {
            'list': [
                # dao, controller
                2, 3
            ],
        },
        {
            'list': [
                # service, controller
                1, 3
            ],
        },
        {
            'list': [
                # service, dao
                1, 2
            ],
        }
    ]
    return md_list
