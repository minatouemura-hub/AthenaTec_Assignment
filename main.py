import argparse
import os
from pathlib import Path

import pandas as pd

from arg import get_args
from eda import EDA
from model import Predictor


def main(args: argparse.Namespace):
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "plt"
    input_file_path = base_dir / "multivariate-time-series-prediction" / "ett.csv"

    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"{input_file_path}にett.csvファイルがありません．")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    data_gen = pd.read_csv(input_file_path)

    # 1.基礎統計などの可視化
    eda = EDA(data_gen=data_gen, target_col="OT", output_path=output_dir)
    eda.basic_statics()
    # 2.前処理+予測
    predictor = Predictor(
        model_args=args, data_gen=data_gen, target_col="OT", output_path=output_dir
    )
    predictor.predict_with_stl()


if __name__ == "__main__":
    args = get_args()
    main(args)
