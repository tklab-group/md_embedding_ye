tarmaq_result_data = {
    # 'tomcat': {
    #     '1': {
    #         'micro': 0.251,
    #         'macro': 0.204
    #     },
    #     '5': {
    #         'micro': 0.418,
    #         'macro': 0.363
    #     },
    #     '10': {
    #         'micro': 0.518,
    #         'macro': 0.447
    #     },
    #     '15': {
    #         'micro': 0.573,
    #         'macro': 0.490
    #     },
    #     '20': {
    #         'micro': 0.603,
    #         'macro': 0.516
    #     },
    # },
    'tomcat': {
        '1': {'micro': 0.257, 'macro': 0.209},
        '5': {'micro': 0.429, 'macro': 0.373},
        '10': {'micro': 0.531, 'macro': 0.455},
        '15': {'micro': 0.578, 'macro': 0.493},
        '20': {'micro': 0.612, 'macro': 0.524}
    },
    # 'cassandra': {
    #     '1': {
    #         'micro': 0.589,
    #         'macro': 0.432
    #     },
    #     '5': {
    #         'micro': 0.720,
    #         'macro': 0.536
    #     },
    #     '10': {
    #         'micro': 0.766,
    #         'macro': 0.578
    #     },
    #     '15': {
    #         'micro': 0.800,
    #         'macro': 0.611
    #     },
    #     '20': {
    #         'micro': 0.820,
    #         'macro': 0.631
    #     },
    # },
    'cassandra': {
        '1': {'micro': 0.592, 'macro': 0.436},
        '5': {'micro': 0.724, 'macro': 0.54},
        '10': {'micro': 0.77, 'macro': 0.581},
        '15': {'micro': 0.803, 'macro': 0.614},
        '20': {'micro': 0.821, 'macro': 0.633}
    },
    # 'hadoop': {
    #     '1': {'micro': 0.221, 'macro': 0.125},
    #     '5': {'micro': 0.369, 'macro': 0.231},
    #     '10': {'micro': 0.469, 'macro': 0.303},
    #     '15': {'micro': 0.525, 'macro': 0.345},
    #     '20': {'micro': 0.565, 'macro': 0.375},
    # },
    'hadoop': {
        '1': {'micro': 0.245, 'macro': 0.137},
        '5': {'micro': 0.38, 'macro': 0.237},
        '10': {'micro': 0.473, 'macro': 0.305},
        '15': {'micro': 0.53, 'macro': 0.348},
        '20': {'micro': 0.568, 'macro': 0.376}
    },
    # 'lucene': {
    #     '1': {'micro': 0.201, 'macro': 0.145},
    #     '5': {'micro': 0.371, 'macro': 0.288},
    #     '10': {'micro': 0.467, 'macro': 0.369},
    #     '15': {'micro': 0.518, 'macro': 0.413},
    #     '20': {'micro': 0.567, 'macro': 0.451},
    # },
    'lucene': {
        '1': {'micro': 0.226, 'macro': 0.157},
        '5': {'micro': 0.386, 'macro': 0.296},
        '10': {'micro': 0.475, 'macro': 0.376},
        '15': {'micro': 0.526, 'macro': 0.418},
        '20': {'micro': 0.574, 'macro': 0.457}
    },
    # 'hbase': {
    #     '1': {'micro': 0.193, 'macro': 0.151},
    #     '5': {'micro': 0.331, 'macro': 0.274},
    #     '10': {'micro': 0.428, 'macro': 0.343},
    #     '15': {'micro': 0.486, 'macro': 0.394},
    #     '20': {'micro': 0.526, 'macro': 0.428},
    # },
    'hbase': {
        '1': {'micro': 0.22, 'macro': 0.165},
        '5': {'micro': 0.353, 'macro': 0.285},
        '10': {'micro': 0.438, 'macro': 0.351},
        '15': {'micro': 0.495, 'macro': 0.4},
        '20': {'micro': 0.532, 'macro': 0.432}
    },
    # 'camel': {
    #     '1': {'micro': 0.331, 'macro': 0.264},
    #     '5': {'micro': 0.511, 'macro': 0.393},
    #     '10': {'micro': 0.599, 'macro': 0.453},
    #     '15': {'micro': 0.644, 'macro': 0.486},
    #     '20': {'micro': 0.684, 'macro': 0.516},
    # },
    'camel': {
        '1': {'micro': 0.355, 'macro': 0.275},
        '5': {'micro': 0.52, 'macro': 0.397},
        '10': {'micro': 0.605, 'macro': 0.457},
        '15': {'micro': 0.651, 'macro': 0.49},
        '20': {'micro': 0.693, 'macro': 0.521}
    }
}


def get_tarmaq_result(git_name):
    return tarmaq_result_data[git_name]
