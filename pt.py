import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from etna.datasets.tsdataset import TSDataset
# from etna.metrics import MAPE
# from etna.metrics import SMAPE
from etna.analysis import plot_forecast
from etna.transforms import LagTransform
from etna.models import CatBoostMultiSegmentModel
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

pd.options.mode.chained_assignment = None
df = pd.read_csv('data/prices_hist.csv')
df["timestamp"] = pd.to_datetime(df["datetime"])
df["target"] = df["price"] 
df.drop(columns=["datetime", "price"], inplace=True)
df["segment"] = "main"
df = TSDataset.to_dataset(df)
ts = TSDataset(df, freq="W-FRI")

def forecasting_plot(df, ts, fr, clr_dn, clr_up, t_1, t_2):
  h = df.shape[0]
  df['label'] = np.NaN

  clr = []
  r_clr = []

  for i in range(0, h - 1):
    if df[fr][i + 1] - df[fr][i] > 0:
      df['label'][i] = 1
      clr.append(clr_up)
    if df[fr][i + 1] - df[fr][i] <= 0:
      df['label'][i] = 0
      clr.append(clr_dn)
  
  plt.figure('Прогноз цен на арматуру', figsize=(64, 12), dpi=80)
  # axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
  # text_box = TextBox(axbox, label="Important")
  # text_box.on_submit(submit)
  # text_box.set_val("5")  # Trigger `submit` with the initial string.
  # fig.subplots_adjust(bottom=0.2)

  def submit(expression):
    """
    Update the plotted function to the new math *expression*.

    *expression* is a string using "t" as its independent variable, e.g.
    "t ** 3".
    """
    # ydata = eval(expression)
    # plt.draw()
    pass

  text_box = TextBox("Evaluate", textalignment="center")
  text_box.on_submit(submit)
  text_box.set_val("5")  # Trigger `submit` with the initial string. 
  r_clr.append("black")
  clr.append("black")
  labels = ['неделя {0}'.format(i) for i in range(1, len(df[fr])+1)]
  for i in range(len(df[fr])):
    plt.scatter(df[ts], df[fr], color=clr,)
  for label, x, y in zip(labels, df[ts], df[fr]):
    plt.annotate(
      label,
      xy=(x, y), xytext=(-20, 20),
      textcoords='offset points', ha='right', va='bottom',
      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
      arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    
  plt.plot(df[ts], df[fr], label='predict')
  plt.legend(fontsize=50, handlelength=4)
  plt.suptitle(t_1, fontsize=40)
  plt.title(t_2, fontsize=30)
  plt.show()

def catBoost(ts, date_train_start, date_train_end, date_test_start, date_test_end, HORIZON):
  train_ts, test_ts = ts.train_test_split(
    train_start=date_train_start,
    train_end=date_train_end,
    test_start=date_test_start,
    test_end=date_test_end,
)

  lags = LagTransform(in_column="target", lags=list(range(1, 94, 1)))
  train_ts.fit_transform([lags])
  model = CatBoostMultiSegmentModel(iterations=1004, depth=6, learning_rate=0.0300007, l2_leaf_reg=2.001, bootstrap_type='MVS')
  model.fit(train_ts)
  future_ts = train_ts.make_future(HORIZON)
  forecast_ts = model.forecast(future_ts)
  train_ts.inverse_transform()

  # test_df = test_ts.to_pandas(True)[['timestamp','target']]
  forecast_df = forecast_ts.to_pandas(True)[['timestamp','target']]

  result_df = forecast_df.copy()
  # result_df = pd.concat([result_df, test_df['target']], axis=1)

  # result_df.columns = ['timestamp', 'forecast', 'target']
  result_df.columns = ['timestamp', 'forecast']

  return result_df

def result(n=5):
    reinforce = catBoost(ts, "2018-01-05", "2022-06-30", "2022-07-01", "2022-12-23", n+10)
    # print(reinforce)
    rf = reinforce.forecast
    text_1 = f"Прогноз на 10 недель, начиная с {n} недели."
    # print(f"Прогноз на 10 недель, начиная с {n} недели.")
    buy=1
    print(n)
    minimal = rf[n]
    print(minimal)
    for i in range(1, 10):
        print(rf[n+i])
        if rf[n+i]>minimal:
            buy +=1
        else:
            break
            # print(minimal)
            # print(i)
            # minimal = rf[n+i]
    text_2 = "Покупать на " + str(buy) + " недель."
    # print("Покупать на ", buy, "недели.")
    forecasting_plot(reinforce, 'timestamp', 'forecast', '#8b00ff', '#9BED00', text_1, text_2)

result()