import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import signal


def time_lag(df, lagged_qty, steps = 1, season = None):

    for step in range(steps):
        df[f't-{step+1}'] = df[lagged_qty].shift(step+1)
    
    if season != None:

        df[f'S({season})'] = df[lagged_qty].shift(season)


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
    if plot_title != None:
        plt.savefig(f'Images/{title}.png')
    
    
    plt.show()
    
def model_test_no_space(df,target,steps=1,split_date = '2023-05-31'):
    df_model_dev = df.copy()
    time_lag(df_model_dev,target,steps=steps)

    df_model_dev = df_model_dev.dropna().reset_index(drop=True)

    df_train = df_model_dev[df_model_dev['Date'] <= split_date]
    df_test = df_model_dev[df_model_dev['Date'] > split_date]

    target = df_train[target]
    features = df_train.filter(like='t-',axis=1)

    features = sm.add_constant(features)

    mlr_model = sm.OLS(target,features).fit()
    
    return (mlr_model, mlr_model.summary())

def acf_pacf(series, title = 'Autocorrelations'):
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8),sharex=True)
    fig.suptitle(title)
    # Make ACF plot
    plot_acf(series.dropna(), lags=10, zero=False, ax=ax1)
    # Make PACF plot
    plot_pacf(series.dropna(), lags=10, zero=False, ax=ax2)

    ax2.set_xlabel('Lags')
    plt.tight_layout()
    if title != "Autocorrelations":
        plt.savefig(f'Images/{title}.png')
    plt.show()

def mean_std_view(df,quantity,title="Stationarity of Data",window = 30):
    df_stat= df.copy()

    df_stat['rolling mean'] = df[quantity].rolling(window=window).mean()
    df_stat['overall_mean'] = df[quantity].mean()

    df_stat['rolling std'] = df[quantity].rolling(window=window).std()
    df_stat['overall_std'] = df[quantity].std()

    tb_blue = '#1f77b4'
    tb_orange = '#ff7f0e'

    fig, axes= plt.subplots(2,1, figsize=(10, 8),sharex=True)

    sns.lineplot(data=df_stat, x='Date',y=quantity,errorbar=None, ax=axes[0])
    sns.lineplot(df_stat,x='Date', y='rolling mean',errorbar=None,ax=axes[1],color=tb_blue)
    sns.lineplot(df_stat,x='Date', y='rolling std',errorbar=None,ax=axes[1],color=tb_orange)
    sns.lineplot(df_stat,x='Date',y='overall_mean',errorbar=None,ax=axes[1],linestyle = ':',color=tb_blue)
    sns.lineplot(df_stat,x='Date',y='overall_std',errorbar=None,ax=axes[1],linestyle = ':',color= tb_orange)
                
    axes[0].set(title = title,
        xlabel = 'Date',
        ylabel='Difference')

    axes[1].set(title = 'Rolling 30 Day Mean and Standard Deviation',
        xlabel = 'Date',
        ylabel='Variation')
    axes[1].legend(loc="upper left", labels=[ "Rolling Mean", "Rolling Std Dev"])

    sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if title != "Stationarity of Data":
        plt.savefig(f'Images/{title}.png')