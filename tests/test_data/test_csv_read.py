# /Users/yejianfeng/Desktop/test_csv_read/2021LabExperiments/analyze_mills_data
import pandas as pd

if __name__ == '__main__':
    print (pd.__version__)
    path_url = '~/Desktop/test_csv_read/2021LabExperiments/analyze_mills_data/data/ReplicationPackage/Results/GAEffectiveness/CompleteQuery/AspectJ.csv'

    query_lists = pd.read_csv(
        path_url,
        skipinitialspace=True,
        dtype={
            "QueryId": str,
            "NumberRelevantDocuments": int,
            "OldRelevantIndices": int,
            "NewRelevantIndices": int,
            "OldQuery": str,
            "OldNonUniqueTerms": int,
            "OldUniqueTerms": int,
            "NewQuery": str,
            "NewNonUniqueTerms": int,
            "NewUniqueTerms": int
        }
    )
    print('query_lists', query_lists)
