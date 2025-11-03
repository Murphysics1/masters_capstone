import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import signal


def time_lag(df, lagged_qty, steps = 1, season = None):

    for step in range(steps):
        df[f't-{step+1}'] = df[lagged_qty].shift(step+1)
    
    if season != None:

        df[f'S({season})'] = df[lagged_qty].shift(season)
    


def spectrum(val, name = None, figsize = (10,4)):
    f, Pxx = signal.periodogram(val)

    fig, ax = plt.subplots(figsize=figsize)

    if name == None:
        title = 'Spectrum Analysis'
    else:
        title = f'Spectrum Analysis of {name}'

    ax.semilogy(f,Pxx)
    ax.set(
        title = title,
        xlabel = 'Frequency',
        ylabel = 'Power Spectrum Density')
    #ax.set_ylim([1e-4,1e1])
    plt.show()


def trend_stationary(df,col,alpha=0.05):

    res = adfuller(df[col])

    print(f'ADF statistic is {res[0]:.2f}')

    if res[1] >= alpha:
        phrase = 'and is therefore not stationary.'
        extent = 'We cannot reject the null hypothesis of a unit root'
    else:
        phrase = 'and is therefore stationary.'
        if res[0] < res[4]['1%']:
            extent = 'We can reject at 99% confidence'
        elif res[0] < res[4]['5%']:
            extent = 'We can reject at 95% confidence'
        elif res[0] < res[4]['10%']:
            extent = 'We can reject at 90% confidence'
        else:
            extent = ''
    
    print(f'the p-value of this series is {res[1]:.2f} {phrase}\n{extent}')

def time_series_decomposition(df, period = 30, plot_title = None, plot_y = None):

    decomp = seasonal_decompose(df,period=period)

    decomp_df = pd.DataFrame({
    'observed': decomp.observed,
    'trend': decomp.trend,
    'seasonal': decomp.seasonal,
    'residual': decomp.resid
     })
    
    if plot_title != None:
        title = plot_title
    else:
        title = "Seasonal Decomposition"

    if plot_y != None:
        y_label = plot_y
    else:
        y_label = "Quantity"

    fig, axes = plt.subplots(4,1, figsize=(10,10),sharex=True)
    fig.suptitle(title)

    line = 0.5

    decomp.observed.plot(ax=axes[0], linewidth = line)
    decomp.trend.plot(ax=axes[1],linewidth = line)
    decomp.seasonal.plot(ax=axes[2],linewidth = line)
    decomp.resid.plot(ax=axes[3],linewidth = line)

    for i in range(4):
        axes[i].set_ylabel(y_label)
        axes[i].set_xlabel('Date')

    axes[0].set_title("Observed")
    axes[1].set_title("Trend")
    axes[2].set_title("Seasonality")
    axes[3].set_title("Residuals")

    plt.tight_layout()
    plt.show()
    