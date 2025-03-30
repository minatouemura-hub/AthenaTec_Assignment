import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

sys.path.append(str(Path(__file__).resolve().parent.parent))
from preprocess import Preprocess  # noqa


class Predictor(Preprocess):
    def __init__(self, model_args, data_gen: pd.DataFrame, target_col: str, output_path: Path):
        Preprocess.__init__(self, data_gen=data_gen, target_col=target_col, output_path=output_path)
        self.predictor_output_path = output_path / "predictor"
        if not os.path.isdir(self.predictor_output_path):
            os.mkdir(self.predictor_output_path)
        self.model_args = model_args

        self.test_ratio = model_args.test_ratio
        total_len = len(self.series)
        self.test_ind = int((1 - self.test_ratio) * total_len)
        self.true = self.series[self.test_ind :]

    def predict_with_stl(self):
        period, result = self.seasonal_grid_search()
        trend = result.trend
        seasonal = result.seasonal
        resid = result.resid

        # 予測結果の保存
        reconstructed_preds = []
        if self.model_args.use_model == "AR":
            use_models = ["AR"]
            reconstructed_pred = self.ar_predict(
                trend=trend, seasonal=seasonal, resid=resid, period=period
            )
            reconstructed_preds.append(reconstructed_pred)
        elif self.model_args.use_model == "ARIMA":
            use_models = ["ARIMA"]
            reconstructed_pred = self.arima_predict(trend, seasonal, resid, period)
            reconstructed_preds.append(reconstructed_pred)
        elif self.model_args.use_model == "MSTL_AR":
            use_models = ["MSTL_AR"]
            reconstructed_preds.append(self.ar_with_mstl(period=period))
        else:
            use_models = ["AR", "ARIMA"]
            reconstructed_preds.append(
                self.ar_predict(trend=trend, seasonal=seasonal, resid=resid, period=period)
            )
            reconstructed_preds.append(self.arima_predict(trend, seasonal, resid, period))
        # 予測結果の可視化
        self._plt_tsa(reconstructed_preds=reconstructed_preds, use_models=use_models)

    def ar_with_mstl(self, period=24):
        trends, seasonals, final_resid = self.multiple_seasonal_decomp()
        train_resid = final_resid[: self.test_ind]
        test_resid = final_resid[self.test_ind :]  # noqa

        ar_model = AutoReg(train_resid, lags=period)
        ar_fit = ar_model.fit()

        pred_resid = ar_fit.predict(start=len(train_resid), end=len(final_resid) - 1)

        reconstructed_pred = 0
        for trend, seasonal in zip(trends, seasonals):
            reconstructed_pred += trend[self.test_ind :] + seasonal[self.test_ind :]
        reconstructed_pred += pred_resid

        # 精度表示
        mse = mean_squared_error(self.true, reconstructed_pred)
        r2 = r2_score(self.true, reconstructed_pred)
        aic = ar_fit.aic
        print("\n == MSTL + ARの精度 == ")
        print(f"MSTL + AR（resid予測）の MSE: {mse:.3f}")
        print(f"MSTL + AR (resid予測）の R² : {r2:.3f}")
        print(f"MSTL + AR（resid予測）の AIC: {aic:.3f}")
        return reconstructed_pred

    def ar_predict(self, trend, seasonal, resid, period):
        train_resid = resid[: self.test_ind]
        test_resid = resid[self.test_ind :]  # noqa

        ar_model = AutoReg(train_resid, lags=period)
        ar_fit = ar_model.fit()

        pred_resid = ar_fit.predict(start=len(train_resid), end=len(resid) - 1)

        trend_tail = trend[self.test_ind :]
        seasonal_tail = seasonal[self.test_ind :]
        reconstructed_pred = trend_tail + seasonal_tail + pred_resid

        # 精度表示
        mse = mean_squared_error(self.true, reconstructed_pred)
        r2 = r2_score(self.true, reconstructed_pred)
        aic = ar_fit.aic
        print("\n == ARの精度 == ")
        print(f"STL + AR（resid予測）の MSE: {mse:.3f}")
        print(f"STL + AR (resid予測）の R² : {r2:.3f}")
        print(f"STL + AR（resid予測）の AIC: {aic:.3f}")

        return reconstructed_pred

    # ARIMA
    def arima_predict(self, trend, seasonal, resid, period):
        train_resid = resid[: self.test_ind]
        test_resid = resid[self.test_ind :]  # noqa
        trend_tail = trend[self.test_ind :]
        seasonal_tail = seasonal[self.test_ind :]

        # AICを用いたパラメータの探索
        model = auto_arima(
            train_resid,
            seasonal=False,
            trace=False,
            stepwise=self.model_args.arima_stepwise,
            max_p=5,
            max_q=5,
            max_d=1,
        )
        arima_model = ARIMA(train_resid, order=model.order)
        model_fit = arima_model.fit()

        # residの予測
        pred_resid = model_fit.predict(start=len(train_resid), end=len(resid) - 1)

        # 元データ予測に再構成
        reconstructed_pred = trend_tail + seasonal_tail + pred_resid

        # 精度の表示
        mse = mean_squared_error(self.true, reconstructed_pred)
        r2 = r2_score(self.true, reconstructed_pred)
        aic = model_fit.aic
        print("\n == ARIMAの精度 == ")
        print(f"STL + ARIMA（resid予測）の MSE: {mse:.3f}")
        print(f"STL + ARIMA（resid予測）の R² : {r2:.3f}")
        print(f"STL + ARIMA（resid予測）の AIC: {aic:.3f}")
        return reconstructed_pred

    def _plt_tsa(self, reconstructed_preds: list, use_models: list):
        # 可視化
        plt.figure(figsize=(12, 5))
        plt.plot(self.true.index, self.true.values, label="実測値", color="blue")
        for model_name, reconst_pred in zip(use_models, reconstructed_preds):
            plt.plot(
                self.true.index,
                reconst_pred.values,
                label=f"予測値（STL + {model_name}）",
                linestyle="--",
            )
        plt.title("resid予測 → STL構成に復元")
        plt.xlabel("時間ステップ")
        plt.ylabel("OT")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        model_str = "_".join(use_models)
        plt.savefig(self.predictor_output_path / f"stl_{model_str}_forecast.jpg")
        plt.close()
