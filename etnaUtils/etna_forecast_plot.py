import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import PySimpleGUI as sg
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from etna.datasets.tsdataset import TSDataset
# from etna.metrics import MAPE
# from etna.metrics import SMAPE
# from etna.analysis import plot_forecast
from etna.transforms import LagTransform
from etna.models import CatBoostMultiSegmentModel
import matplotlib.pyplot as plt
# from matplotlib.widgets import TextBox
df = pd.read_csv('data/prices_hist.csv')
df["timestamp"] = pd.to_datetime(df["datetime"])
df["target"] = df["price"] 
df.drop(columns=["datetime", "price"], inplace=True)
df["segment"] = "main"
df = TSDataset.to_dataset(df)
ts = TSDataset(df, freq="W-FRI")
def main():
  matplotlib.use('TkAgg')

  pd.options.mode.chained_assignment = None
  

  layout = [
    [sg.Text('С какой недели прогнозировать: ', background_color='#8DC2FC'), sg.InputText(key='-INPUT-'), sg.Submit("Ввод")],
    [sg.Canvas(size=(640, 320), key='-CANVAS-')],
    [sg.Cancel("Выход")]
  ]

  window = sg.Window('Прогноз цен на арматуру', layout, finalize=True, element_justification='center', background_color='#8DC2FC')
  #draw_figure(window['-CANVAS-'].TKCanvas, result(1))
  figure_agg = None
  v = 1


  while True:
      event, values = window.read()
      if event in (None, 'Exit', 'Выход'):
          break

      if figure_agg:
          # ** IMPORTANT ** Clean up previous drawing before drawing again
          delete_figure_agg(figure_agg)

      if event == 'Ввод':
          v = int(values['-INPUT-'])

      figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, result(v))
# Draw function
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
 
  plt.figure('Прогноз цен на арматуру', figsize=(20, 8), dpi=80)
  r_clr.append("black")
  clr.append("#B2B2B2")
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
  plt.suptitle(t_1, fontsize=24)
  plt.title(t_2, fontsize=18)
  return plt.gcf()
  # plt.show()

# Dataset and model function
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

# Main function for start
def result(n=5):
    reinforce = catBoost(ts, "2018-01-05", "2022-06-30", "2022-07-01", "2022-12-23", n+10)
    # print(reinforce)
    rf = reinforce.forecast
    text_1 = f"Прогноз на 10 недель, начиная с {n}. \n"
    # print(f"Прогноз на 10 недель, начиная с {n} недели.")
    buy=1
    # print(n)
    # minimal = rf[n]
    # print(minimal)
    # for i in range(1, 10):
        # print(rf[n+i])
        # if rf[n+i]>minimal:
            # buy +=1
        # else:
            # break
            # print(minimal)
            # print(i)
            # minimal = rf[n+i]
    for i in range(9):
        if rf[n - 1 + i] <= rf[n - 1 + i + 1]:
            buy += 1
        else:
            break
    text_2 = "Покупать на " + str(buy) + " нед."
    # print("Покупать на ", buy, "недели.")
    return forecasting_plot(reinforce, 'timestamp', 'forecast', '#DB133B', '#FFCC00', text_1, text_2)


def draw_figure(canvas, figure):
    if not hasattr(draw_figure, 'canvas_packed'):
        draw_figure.canvas_packed = {}
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    widget = figure_canvas_agg.get_tk_widget()
    if widget not in draw_figure.canvas_packed:
        draw_figure.canvas_packed[widget] = figure
        widget.pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    try:
        draw_figure.canvas_packed.pop(figure_agg.get_tk_widget())
    except Exception as e:
        print(f'Error removing {figure_agg} from list', e)
    plt.close('all')




if __name__ == '__main__':
    main()