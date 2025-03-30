import os
from pathlib import Path

import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import acf, adfuller  # noqa


class EDA:
    def __init__(self, data_gen: pd.DataFrame, output_path: Path, target_col: str = "OT"):
        self.data_gen = data_gen
        self.target_col = target_col
        self.eda_output_path = output_path / "eda"

        if not os.path.isdir(self.eda_output_path):
            os.mkdir(self.eda_output_path)

    def explanatoty_data_analysis(self):
        self.basic_statics()
        self.check_stationary()
        self.corr_headmap()
        self.fft_spectrum()

    # 基礎統計の確認
    def basic_statics(self):
        print(f"==Basic Statics:{self.target_col}==")
        print(self.data_gen[self.target_col].describe())

    # 定常性の検定
    def check_stationary(self):
        series = self.data_gen[self.target_col].dropna()
        result = adfuller(series)
        print(f"\n==ADF 検定結果 for {self.target_col}==")
        print(f"Test Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        print(f"# Lags Used: {result[2]}")
        print(f"# Observations Used: {result[3]}")
        print("Critical Values:")
        for key, value in result[4].items():
            print(f"   {key}: {value}")

        if result[1] < 0.05:
            print("\n この系列は定常とみなせます（p < 0.05）")
        else:
            print("\n この系列は非定常の可能性があります（p >= 0.05）")

    # 他の変数との関係
    def corr_headmap(self):
        corr_matrix = self.data_gen.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
        plt.title("相関行列(ETTデータ)")
        plt.savefig(self.eda_output_path / "corr_heatmap.jpg")
        plt.close()

    # 周期性があるかの確認
    def fft_spectrum(self, sampling_rate: float = 1 / 60, top_n: int = 10):
        series = self.data_gen[self.target_col].dropna().values
        n = len(series)

        fft_result = np.fft.fft(series)
        fft_freq = np.fft.fftfreq(n, d=sampling_rate)

        spectrum = np.abs(fft_result[: n // 2])
        freq = fft_freq[: n // 2]

        # DC成分を除く（0Hzの定数成分）
        freq = freq[1:]
        spectrum = spectrum[1:]

        # 上位N個のピークを抽出
        top_indices = np.argsort(spectrum)[-top_n:][::-1]
        top_freqs = freq[top_indices]
        top_powers = spectrum[top_indices]
        top_periods = 1 / top_freqs  # 周期に変換

        # 結果表示
        print("== 主な周期性 ==")
        for i in range(top_n):
            print(
                f"{i+1}. 周期: {top_periods[i]:.2f} 時間（周波数: {top_freqs[i]:.5f} Hz, パワー: {top_powers[i]:.2f}）"  # noqa
            )

        # 周期ベースの棒グラフ（横軸: 周期）
        plt.figure(figsize=(10, 5))
        plt.bar(top_periods, top_powers, width=3.0)
        plt.xlabel("周期（時間単位）")
        plt.ylabel("パワー（強さ）")
        plt.title(f"{self.target_col} の主要な周期性（Top {top_n}）")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.eda_output_path / "fft_top_peaks.jpg")
        plt.close()

    # 一応自己相関も調べる
    def plt_acf(self):
        acf_vals = acf(self.data_gen["OT"].dropna(), nlags=200)
        plt.plot(acf_vals)
        plt.title("自己相関関数（ACF）")
        plt.xlabel("ラグ（時間）")
        plt.ylabel("自己相関")
        plt.grid(False)
        plt.savefig(self.eda_output_path / "acf.jpg")
        plt.close()
