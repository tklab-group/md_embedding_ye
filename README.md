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

# Mongodbに保存した実験データのVersionのリスト
 - data/git_name_version.py