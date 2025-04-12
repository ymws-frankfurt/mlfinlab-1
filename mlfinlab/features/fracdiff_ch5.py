import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getWeights(d, size):
    # Compute weights for fractional differencing.
    # A small threshold in practice can later be applied to drop insignificant weights.
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] * (d - k + 1) / k
        w.append(w_)
    # Reverse the list so that the weights are in chronological order and return as a column vector.
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

# Optional: a plotting routine to visualize the weight decay for various d values.
def plotWeights(dRange, nPlots, size):
    w_df = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w = getWeights(d, size=size)
        # Create a DataFrame with the reversed index for plotting.
        w_temp = pd.DataFrame(w, index=range(w.shape[0])[::-1], columns=[d])
        w_df = w_df.join(w_temp, how='outer')
    ax = w_df.plot()
    ax.legend(loc='upper left')
    plt.show()


def fracDiff(series, d, thres=0.01):
    '''
    Compute the fractionally differentiated series using an expanding window.
    For thres=1, nothing is skipped. d can be any positive fractional value.
    '''
    # 1) Compute weights for the full series
    weights = getWeights(d, series.shape[0])
    
    # 2) Determine the number of initial observations to skip based on weight-loss threshold
    weights_cum = np.cumsum(np.abs(weights))
    weights_cum /= weights_cum[-1]
    skip = weights_cum[weights_cum > thres].shape[0]
    
    # 3) Apply weights to the series to compute the fractionally differenced values
    output_df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        output_series = pd.Series(index=series.index, dtype='float64')
        for iloc in range(skip, series_f.shape[0]):
            loc = series_f.index[iloc]
            output_series[loc] = np.dot(weights[-(iloc + 1):, :].T, series_f.loc[:loc])[0, 0]
        output_df[name] = output_series.copy(deep=True)
    output_df = pd.concat(output_df, axis=1)
    return output_df


def getWeights_FFD(d, thres):
    '''
    Compute weights for the fixed-width window fractional differencing.
    The window is truncated when an individual weight falls below thres.
    '''
    weights = [1.]
    k = 1
    while True:
        w_next = -weights[-1] * (d - k + 1) / k
        if np.abs(w_next) < thres:
            break
        weights.append(w_next)
        k += 1
    # Reverse the weights so that the first element corresponds to the oldest observation
    weights = np.array(weights[::-1]).reshape(-1, 1)
    return weights

def fracDiff_FFD(series, d, thres=1e-5):
    '''
    Compute the fractionally differentiated series using a fixed-width window.
    thres determines the cutoff for the weights.
    '''
    # 1) Compute weights using the fixed window approach
    weights = getWeights_FFD(d, thres)
    width = len(weights) - 1
    
    # 2) Apply weights to the series
    output_df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        output_series = pd.Series(index=series.index, dtype='float64')
        for iloc in range(width, series_f.shape[0]):
            loc0 = series_f.index[iloc - width]
            loc1 = series_f.index[iloc]
            output_series[loc1] = np.dot(weights.T, series_f.loc[loc0:loc1])[0, 0]
        output_df[name] = output_series.copy(deep=True)
    output_df = pd.concat(output_df, axis=1)
    return output_df


from statsmodels.tsa.stattools import adfuller

def plotMinFFD():
    from statsmodels.tsa.stattools import adfuller
    import matplotlib.pyplot as plt
    
    path, instName = './', 'ES1_Index_Method12'
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    df0 = pd.read_csv(path + instName + '.csv', index_col=0, parse_dates=True)
    
    for d in np.linspace(0, 1, 11):
        # Downcast to daily observations and take the log-prices
        df1 = np.log(df0[['Close']]).resample('1D').last()
        df1.dropna(inplace=True)
        
        # Apply the fixed-window fractional differentiation
        df2 = fracDiff_FFD(df1, d, thres=0.01)
        # Compute correlation between the original and the differenced series
        corr = np.corrcoef(df1.loc[df2.index, 'Close'], df2['Close'])[0, 1]
        
        # Apply the ADF test
        adf_result = adfuller(df2['Close'], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(adf_result[:4]) + [adf_result[4]['5%']] + [corr]
    
    # Save results and plot the ADF statistic and correlation
    out.to_csv(path + instName + '_testMinFFD.csv')
    ax = out[['adfStat', 'corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    plt.savefig(path + instName + '_testMinFFD.png')
    plt.show()
    return



if __name__ == '__main__':
    plotWeights(dRange=[0, 1], nPlots=11, size=6)
    plotWeights(dRange=[1, 2], nPlots=11, size=6)
