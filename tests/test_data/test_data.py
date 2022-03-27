def get_method_map():
    method_map = [
        # {
        #     'item': 'src/tklab/hagward/lcextractor/cli/DBHandler#public_void_storeRepositoryReaderResult(RepositoryReaderResult_result,String_dbName)',
        #     'index': 4
        # },
        # {
        #     'item': 'src/tklab/hagward/lcextractor/scm/MappedList#public_void_setReplaced(T_oldItem,T_newItem)',
        #     'index': 3
        # },
        # {
        #     'item': 'src/tklab/hagward/lcextractor/scm/Commit#public_Commit(RevCommit_revCommit)',
        #     'index': 2
        # },
        # {
        #     'item': 'src/tklab/hagward/lcextractor/LCExtractor#public_LCExtractor()',
        #     'index': 1
        # },
        # {
        #     'item': 'src/tklab/hagward/lcextractor/coupling/Predictor#public_RepositoryReaderResult_getData()',
        #     'index': 0
        # },
        # {
        #     'item': 'src/tklab/hagward/lcextractor/cli/DBHandler#public_void_storeRepositoryReaderResult(RepositoryReaderResult_result,String_dbName)',
        #     'index': 4
        # },
        {
            'item': 'src/tklab/hagward/cli/DBHandler#public_void_storeRepositoryReaderResult(RepositoryReaderResult_result,String_dbName)',
            'index': 4
        },
        {
            'item': 'src/tklab/hagward/scm/MappedList#public_void_setReplaced(T_oldItem,T_newItem)',
            'index': 3
        },
        {
            'item': 'src/tklab/hagward/scm/Commit#public_Commit(RevCommit_revCommit)',
            'index': 2
        },
        {
            'item': 'src/tklab/hagward/LCExtractor#public_LCExtractor()',
            'index': 1
        },
        {
            'item': 'src/tklab/hagward/coupling/Predictor#public_RepositoryReaderResult_getData()',
            'index': 0
        }
    ]
    return method_map


def get_module_data():
    md_list = [
        {
            'list': [
                4, 3, 2
            ],
        },
        {
            'list': [
                3, 2
            ],
        },
        {
            'list': [
                0, 1, 2
            ],
        },
        {
            'list': [
                1, 2, 3
            ],
        },
        {
            'list': [
                1, 2, 3, 4, 0
            ],
        },
        {
            'list': [
                1, 2
            ],
        }
    ]
    return md_list
