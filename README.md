<p align="center">
  <img src="https://github.com/user-attachments/assets/f46b98f5-b1c2-41ad-9617-a95501a5c266" >
</p>

# オイル温度の時系列予測　<br> -Athena Technologies インターンシップ課題-
## 課題の概要
本課題は[ETDataset](https://github.com/zhouhaoyi/ETDataset)を用いてオイル温度(Oil Temperature)の時系列予測を目的としています．<br>
ETDataset(Electricity Transformer Dataset)電力トランスフォーマの運用中に収集された時系列データを提供するリポジトリです。<br>
このデータセットには、電力トランスフォーマの様々なパラメータが含まれており、その中でもオイル温度に関連するデータは、トランスフォーマの状態を監視し、故障の予測やメンテナンスの最適化に利用されます。

## 使用可能なモデル

### モデルタイプの概要
| モデル | 説明 |
|--------|------|
| `AR` | 残差系列をAR（自己回帰）で予測し、トレンド+季節性と再構成 |
| `ARIMA` | ARIMAで残差系列を予測（auto_arimaによるorder探索） |
| `MSTL_AR` | 複数の周期に基づくSTL分解（マルチシーズナリティ）+ ARモデル |
| `COMPARE` | AR と ARIMA の両方を実行し、精度比較を行う |

## 使用方法
### 1. 環境構築
以下のコマンドで必要なライブラリをインストールします：
```bash
pip install -r requirements.txt
```
ETTデータセット（ett.csvなど）をmultivariate-time-series-prediction/ ディレクトリに配置してください。
もしくは公式リポジトリ [ETDataset](https://github.com/zhouhaoyi/ETDataset) からダウンロードしてください。

### 2. 予測モデルの実行方法
```bash
pyhon main.py --use_model "使用モデル"　--test_ratio "テストデータの割合"
```
例えば以下のコマンドによって，174時間分のデータを評価データとしてMSTL+ARモデルの予測が行われる．
```bash
python main.py --use_model MSTL_AR --test_ratio 0.01
```
![Image](https://github.com/user-attachments/assets/83bd2c49-956d-43ac-bf23-569a50279e49)

### 🔧 コマンドライン引数一覧
以下に実行時に指定可能なコマンドライン引数の一覧を示します．
| 引数名 | デフォルト値 | 型 | 説明 |
|--------|--------------|-----|------|
| `--use_model` | `"ARIMA"` | str | 使用する予測モデルを指定します。選択肢：`AR`, `ARIMA`, `MSTL_AR`, `COMPARE` |
| `--arima_stepwise` | `True` | bool | ARIMAの自動パラメータ選択（`auto_arima`）でstepwise探索を使うかどうか |
| `--test_ratio` | `0.1` | float | 学習データに対するテストデータの割合（例：0.1 → 10%がテストデータ） |
| `--target_col` | `"OT"` | str | 予測対象となるカラム名（例：ETTデータセットの「OT（油温度）」） |

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

