import argparse
from dataclasses import dataclass, field


@dataclass
class MSTLConfig:
    seasonals: list = field(default_factory=lambda: [24, 145, 168])


def get_args():
    parser = argparse.ArgumentParser(description="時系列分析用引数")

    parser.add_argument(
        "--use_model", type=str, default="ARIMA", choices=["AR", "ARIMA", "MSTL_AR", "COMPARE"]
    )
    parser.add_argument("--arima_stepwise", type=bool, default=True)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--target_col", type=str, default="OT")

    return parser.parse_args()
