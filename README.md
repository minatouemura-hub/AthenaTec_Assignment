<image src="https://github.com/user-attachments/assets/f46b98f5-b1c2-41ad-9617-a95501a5c266" >

# オイル温度の時系列予測　<br> -Athena Technologies インターンシップ課題-
## 課題の概要
本課題は[ETDataset](https://github.com/zhouhaoyi/ETDataset)を用いてオイル温度(Oil Temperature)の時系列予測を目的としています．<br>
ETDataset(Electricity Transformer Dataset)電力トランスフォーマの運用中に収集された時系列データを提供するリポジトリです。このデータセットには、電力トランスフォーマの様々なパラメータが含まれており、その中でもオイル温度に関連するデータは、トランスフォーマの状態を監視し、故障の予測やメンテナンスの最適化に利用されます。
## 使用方法
### 1. 環境構築
以下のコマンドで必要なライブラリをインストールします：
```bash
pip install -r requirements.txt
```
ETTデータセット（ett.csvなど）をmultivariate-time-series-prediction/ ディレクトリに配置してください。
もしくは公式リポジトリ [ETDataset](https://github.com/zhouhaoyi/ETDataset) からダウンロードしてください。

###2. 予測モデルの実行
```bash
pyhon main.py --use_model "使用モデル"　--test_ratio "テストデータの割合"
```
例えば...元データの8割をtrain_dataとしてARモデルの予測を行いたい時は

## ディレクトリ構成
```
📦 your-project-name/
├── 📁 multivariate-time-series-prediction/ # データ関連
│   |--  ett.csv          # ETTデータ（元データ）
│
├──  eda/           # データ前処理・STL分解など
│   ├-- data_analysis.py #具体的な処理
│   |--__init__.py
│
├──  preprocess/   #STLを用いた特徴量生成
|    |-- preprocess.py #特徴量生成
|    |-- __init__.py
|
|--  model /
|    |-- models.py  #　モデルの管理(ARモデルやARIMA等)
|    |-- __init__.py
|-- main.py
|-- requirement.txt
|-- arg.py #引数の一括管理
|-- README.md
```

