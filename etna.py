import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from etna.datasets.tsdataset import TSDataset
from etna.metrics import MAPE
from etna.metrics import SMAPE
from etna.analysis import plot_forecast
from etna.transforms import LagTransform
from etna.models import CatBoostMultiSegmentModel
import matplotlib.pyplot as plt
df = pd.read_csv('data/prices_hist.csv')
df["timestamp"] = pd.to_datetime(df["datetime"])
df["target"] = df["price"]
df.drop(columns=["datetime", "price"], inplace=True)
df["segment"] = "main"
df = TSDataset.to_dataset(df)
ts = TSDataset(df, freq="W-FRI")
ts.plot(figsize=(48, 24))
HORIZON = 26
mape = MAPE()
smape = SMAPE()
train_ts, test_ts = ts.train_test_split(
    train_start="2018-01-05",
    train_end="2022-06-30",
    test_start="2022-07-01",
    test_end="2022-12-23",
)
lags = LagTransform(in_column="target", lags=list(range(1, 94, 1)))
train_ts.fit_transform([lags])
model = CatBoostMultiSegmentModel(task_type='CPU')
model.fit(train_ts)
future_ts = train_ts.make_future(HORIZON)
forecast_ts = model.forecast(future_ts)
train_ts.inverse_transform()
plot_forecast(forecast_ts, test_ts, train_ts, n_train_samples=261)
print(mape(y_true=test_ts, y_pred=forecast_ts)['main'])
print(smape(y_true=test_ts, y_pred=forecast_ts)['main'])
plot_forecast(forecast_ts, test_ts, train_ts, n_train_samples=12)
test_df = test_ts.to_pandas(True)[['timestamp','target']]
forecast_df = forecast_ts.to_pandas(True)[['timestamp','target']]
# plt.figure(figsize=(32, 12))
plt.plot(test_df['timestamp'], test_df['target'], label='test' )
plt.plot(forecast_df['timestamp'], forecast_df['target'], color='red', label='predict')

plt.title('Target by Date')
plt.xlabel('Date')
plt.ylabel('Target')

plt.legend()

plt.show() 
result_df = test_df.copy()
result_df = pd.concat([result_df, forecast_df['target']], axis=1)
result_df.columns = ['timestamp', 'target', 'forecast']
def forecast_plot(df, ts, tg, fr, clr_dn, clr_up):
  df['label'] = np.NaN

  clr = []
  r_clr = []

  for i in range(0, len(df[tg]) - 1):
    if df[tg][i + 1] - df[tg][i] > 0:
      r_clr.append(clr_up)
    if df[tg][i + 1] - df[tg][i] <= 0:
      r_clr.append(clr_dn)

  for i in range(0, len(df[tg]) - 1):
    if df[fr][i + 1] - df[fr][i] > 0:
      df['label'][i] = 1
      clr.append(clr_up)
    if df[fr][i + 1] - df[fr][i] < 0:
      df['label'][i] = 0
      clr.append(clr_dn)
      
  plt.figure(figsize=(64, 12), dpi=80)

  r_clr.append("black")
  clr.append("black")
  for i in range(len(df[tg])):
    plt.scatter(df[ts], df[tg], color=r_clr)
    plt.scatter(df[ts], df[fr], color=clr)
  plt.plot(df[ts], df[tg], label='test')
  plt.plot(df[ts], df[fr], label='predict')
  plt.legend(fontsize=50, handlelength=4)
  def result(n):
    result_df = result_df[:n]
    forecast_plot(result_df, 'timestamp', 'target', 'forecast', '#8b00ff', '#9BED00')
    rf = result_df.forecast
    print(f"Прогноз на {n} недель.")
    print("Минимальная цена: ", min(rf))
    print("Покупать на ", rf.loc[result_df.forecast == (min(rf))].index[0], "неделе.")
    result(28)