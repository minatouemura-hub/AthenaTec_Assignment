import argparse


def get_args():
    parser = argparse.ArgumentParser(description="時系列分析用引数")

    parser.add_argument("--use_model", type=str, default="ARIMA", choices=["AR", "ARIMA", "BOTH"])
    parser.add_argument("--arima_stepwise", type=bool, default=True)
    parser.add_argument("--distance", type=str, default="DTW", choices=["DTW", "Euclidean"])

    return parser.parse_args()
