import os
from pathlib import Path

import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

# 1. residual=>ARやLSTM　+ seasonalとtrendは加算
# 2. 状態空間モデルで予測


class Preprocess:
    def __init__(self, data_gen: pd.DataFrame, output_path: Path, target_col: str = "OT"):
        self.preprocess_output_path = output_path / "preprocess"
        if not os.path.isdir(self.preprocess_output_path):
            os.mkdir(self.preprocess_output_path)
        self.series = data_gen[target_col]
        self.target_col = target_col

    def seasonal_grid_search(self):
        periods = [24, 145, 168]
        mse_list = []
        result_list = []
        min_mse = 10000
        for period in periods:
            mse, result = self.stl_decompose(period)
            mse_list.append(mse)
            result_list.append(result)

            if mse < min_mse:
                min_mse = mse
        min_ind = np.argmin(mse_list)
        print(f"周期{periods[min_ind]}のMSE：{mse_list[min_ind]:.3f} が最小でした．")
        return periods[min_ind], result_list[min_ind]

    def reconstruct_error(self, result):
        reconst = result.trend + result.seasonal
        error = self.series - reconst
        mse = np.mean(error**2)
        return mse

    def stl_decompose(self, period=145):
        stl = STL(self.series, period=period)
        result = stl.fit()

        mse = self.reconstruct_error(result=result)

        trend = result.trend
        seasonal = result.seasonal
        resid = result.resid

        # 可視化
        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={"hspace": 0.3})

        axs[0].plot(self.series, color="black")
        axs[0].set_title("OT（元データ）", fontsize=12)

        axs[1].plot(trend, color="tab:blue")
        axs[1].set_title("Trend（トレンド）", fontsize=12)

        axs[2].plot(seasonal, color="tab:green")
        axs[2].set_title("Seasonal（周期性）", fontsize=12)
        axs[2].set_ylim(seasonal.min() * 1.1, seasonal.max() * 1.1)  # 拡大表示

        axs[3].plot(resid, color="tab:red")
        axs[3].set_title("Residual（残差）", fontsize=12)

        for ax in axs:
            ax.grid(True)

        fig.suptitle(f"{self.target_col} のSTL分解（周期: {period} 時間）", fontsize=16)
        plt.xlabel("時間ステップ")
        plt.savefig(self.preprocess_output_path / f"stl_custom_period_{period}.jpg")
        plt.close()
        return mse, result
