import sys
sys.path.append('../../')
from model.ranking_eval import ranking

if __name__ == '__main__':
    predict_word_list = [
        'package.Class#public_int_methodName0()',
        'package.Class3#public_int_methodName1()',
        'package.Class4#public_int_methodName2()',
        'package.Class1#public_int_methodName3()',
        'package.Class2#public_int_methodName4()',
        'package.Class1#public_int_methodName5()',
        'package.Class2#public_int_methodName6()',
        'package.Class3#public_int_methodName8()',
        'package.Class3#public_int_methodName9()',
        'package.Class2#public_int_methodName10()',
        'package.Class2#public_int_methodName11()',
        'package.Class1#public_int_methodName12()',
        'package.Class1#public_int_methodName13()',
        'package.Class#public_int_methodName14()',
    ]
    predict_word_prob_list = []
    predict_file_list = [
        'package.Class11',
        'package.Class22',
        'package.Class33',
        'package.Class4',
    ]
    predict_file_prob_list = []

    result = ranking(predict_word_list, predict_word_prob_list, predict_file_list, predict_file_prob_list)
    for i in range(len(result)):
        print(result[i])
