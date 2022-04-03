# 実験を実行
 - main/normal_multi.py →具体的なパラメータの配置はコードの変数名を見てください
 - tsubame_script/single_project_run.py

# Tsubameで扱うScript
 - tsubame_script/**_normal_##.sh
 - tsubame_script/**_subword_##.sh

# config
 - config/config_default.py
 - zenzeroで実験する時はis_load_data_from_pklをFalseにする、Mongdbからデータを読み込み、実験結果はMongodbに出力する
 - Tsubameで実験する時はis_load_data_from_pklをTrueにする、それは予めにMongodbのデータをPklの形でpkl/**に保存する、実験結果はPklに出力する
 - data/repo_data.py → レポジトリのリストConfig

# pklファイルの生成
 - Tsubameで実験をするためには、予めにpkl/**にはmetadataが必要がある
 - metadataとはmodule data,method map, rename chain, delete recordのことである
 - pklファイルを生成流れとして、Mongodbからデータを読み込み、そのままpklファイルになる
 - pklファイルを生成Scriptは main/to_pkl.py

# Python環境
 - Python 3.8.3

# 環境
 - torch==1.10.1
 - pymongo==4.0.1
 - numpy==1.19.5
 - matplotlib==3.3.4
 - pickle5==0.0.11

# Zenzero GPU環境ためのtorchバージョン
```
 pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
 
# グリッドサーチによるハイパーパラメータの最適化
## グリッドサーチ
 - main/find_best_param.py
 - main/find_best_preprocess.py
 - main/find_best_seed.py
 - main/find_important_part.py
## 最適ハイパーパラメータの確認
 - main/check_best_param.py
 - main/check_best_param2.py
 - main/check_best_preprocess.py
 - main/check_best_preprocess_seed.py
 - main/check_best_seed.py

# RQ精度確認
## RQ1
 - model/eval_database.py →　各Gitレポジトリの各Seedにより実験結果を確認、このScriptはただデータベースのデータを読み込み、そして表示する

## RQ2 and RQ3の関連するソースコード
 - data/low_freq_trace.py →　新規ファイル、あんまり変更されていないファイルと古いファイルへの追跡
 - co_change/low_freq_eval.py →　相関ルールマイニングへの新規ファイル、あんまり変更されていないファイルと古いファイルへの追跡
 - model/eval_database.pyのlow_freq_main →　予測データから追跡データへの生成
 - main/check_low_freq_contexts_database.py →　Contextsが新規ファイル、あんまり変更されていないファイルと古いファイル時の精度確認
 - main/check_low_freq_database.py →　Targetがあんまり変更されていないファイル時の精度確認

# 相関ルールマイニング
 - 精度の確認 co_change/main.py → tomcat_falseの意味はtomcatと直近5000コミットの意味、逆にtomcat_trueとはtomcatと固定5000コミットの意味

# Mongodbに保存した実験データのVersionのリスト
 - data/git_name_version.py
 - Word2vecモデルの予測結果はMongodbのpredict表に保存している
 - 過去に異なるパラメータで実験した予測結果もあるので、管理しやすいためにVersionを使っている
 - 例えば、2022_1_8_tomcat_tomcat_6の意味は、2022年1月8日でTomcatのSeedが6時のデータである
 
