# -*- coding: utf-8 -*-
"""
Author: Andrew Killeen
email: killeenandy@gmail.com
Last edited: 09/08/2023

Script to update stock data if necessary and analyse correlations between stock
pairs if desired. The script then generates a GUI to select parameters for pairs
trading strategy, which can then be backtested or parameters optimised if desired
"""

import pandas as pd
import tkinter as tk
from Optimisation import Optimise
from PairTradeBacktesting import TradingStrategy
import matplotlib.pyplot as plt
from tabulate import tabulate
from DataLoader import LoadData
from EvaluateTradePairs import EvaluateTradePairs

def TradePreprocessing(redownload=False,update_data=False,eval_pairs=False):
    """
    Get stock data, either updated or using what is currently saved, and analyse
    pair correlation if desired.

    Parameters
    ----------
    redownload : Whether to redownload all stock data. The default is False.
    update_data : Whether to update stock data or use what is currently
    downlaoded. The default is False.
    eval_pairs : Whether to evaluate pair correlation metrics before choosing 
    pair to back test. The default is False.

    Returns
    -------
    stock_data : Price data for all stocks
    pair_metrics : Pair correlation metrics (if requested)

    """
    #Pair evaluation parameters    
    period = 2              #Time period over which to analyse pairs
    corr_lo = 0.95          #Lower bound on linear correlation between pairs
    corr_method = 'pearson' #Method to determine linear correlation
    beta = [0.5,3]          #Desired range of regression slope between pairs 
    adf_hi = 0.001          #Upper bound on ADF stationarity test p-val
    hurst_hi = 0.4          #Upper bound on acceptable hurst exponent
    hlife = [3,15]          #Desired range of halflife of mean reversion time
    pair_parameters = [period,corr_lo,corr_method,beta,adf_hi,hurst_hi,hlife]
    
    #Get stock data
    if update_data:
        stock_data = LoadData(redownload_all = redownload, years = 10)
    else:
        stock_data = pd.read_pickle("stock_data.pkl")
    
    #Evaluate pair correlation metrics if desired
    if eval_pairs:        
        pair_metrics = EvaluateTradePairs(pair_parameters,stock_data)
    else:
        pair_metrics = None
    return stock_data, pair_metrics

def PerformBacktest():
    """
    Backtest strategy with desired parameter values, plot salient strategy results
    and display performance metrics
    
    """
    #Parse strategy parameers
    if m.get() == 1:
        method = 'ols'
    elif m.get() == 0:
        method = 'kalman'
    parameters = [w1.get(),w2.get(),w3.get(),w4.get(),w5.get(),w6.get(),w7.get()]
    tickers = [e1.get(),e2.get()]
    pair_data = stock_data[tickers]
    strat_duration = float(e3.get())
    init_equity = int(e4.get())
    order_qty = int(e5.get())
    #Perform back test
    cum_ret,draw_down,slope,overall_metrics,trade_metrics = \
        TradingStrategy(method,parameters,init_equity,order_qty,pair_data,strat_duration)
    
    #Plot performance
    plt.plot(cum_ret[int(parameters[5]):])
    plt.xlabel('Elapsed Time (days)')
    plt.ylabel('Cumulative Returns')
    plt.show()
    plt.plot(draw_down[int(parameters[5]):],'r')
    plt.xlabel('Elapsed Time (days)')
    plt.ylabel('Drawdown')
    plt.show()
    plt.plot(slope[int(parameters[5]):])
    plt.xlabel('Elapsed Time (days)')
    plt.ylabel('Hedge Ratio')
    plt.show()
    #Display strategy metrics
    print('\nStrategy Metrics:')
    print('\n'+tabulate(overall_metrics,headers=['Metric','Value'], floatfmt=".3f", numalign="right"))
    print('\nTrade Metrics:')
    print('\n'+tabulate(trade_metrics,headers=['Metric','Value'], floatfmt=".3f", numalign="right"))
    
    return None
    
def PerformOptimisation():
    """
    Use the DyCORsalgorithm to optimise strategy entry/exit parameters for a given
    pair stocks, along with time period and initial equity. The entry/exit 
    parameter range are the same as the limits of the sliders on the GUI. 
    
    Output the optimised parameters and strategy performance for these parameters.
    
    N.B.:
    1 - As Kalman method is much faster than OLS, it is recommended to use this
    method when optimising parameters.
    
    2 - The outputted parameters are not necessarily 'the optimal' parameters,
    merely a set of particularly high performing parameters for the pair of stocks
    and time period used. Strategy performance may be sensitive to the values outputted.

    """
    #Parse strategy inputs
    if m.get() == 1:
        method = 'ols'
    elif m.get() == 0:
        method = 'kalman'
    tickers = [e1.get(),e2.get()]
    pair_data = stock_data[tickers]
    strat_duration = float(e3.get())
    init_equity = int(e4.get())
    order_qty = int(e5.get())
    #Perform optimisation
    opt_params,cum_ret,draw_down,slope,overall_metrics,trade_metrics = \
        Optimise(method,init_equity,order_qty,pair_data,strat_duration)
    
    #Plot optimal performance
    opt_window = opt_params[5][1]
    plt.plot(100*cum_ret[int(opt_window):])
    plt.xlabel('Elapsed Time (days)')
    plt.ylabel('Cumulative Returns %')
    plt.show()
    plt.plot(100*draw_down[int(opt_window):],'r')
    plt.xlabel('Elapsed Time (days)')
    plt.ylabel('Drawdown %')
    plt.show()
    plt.plot(slope[int(opt_window):])
    plt.xlabel('Elapsed Time (days)')
    plt.ylabel('Hedge Ratio')
    plt.show()
    #Display optimal performance and parameters
    tab_head = ['Parameter','Value']
    print('\nStrategy Metrics:')
    print('\n'+tabulate(overall_metrics,headers=['Metric','Value'], floatfmt=".3f", numalign="right"))
    print('\nTrade Metrics:')
    print('\n'+tabulate(trade_metrics,headers=['Metric','Value'], floatfmt=".3f", numalign="right"))
    print('\nOptimised Parameters:')
    print('\n'+tabulate(opt_params,headers=tab_head, floatfmt=".3f", numalign="right"))

    return None

if __name__ == '__main__':
    #Get stock data and pair analysis (if desired)
    stock_data, pair_metrics = \
        TradePreprocessing(redownload=False,update_data=False,eval_pairs=False)
        
    #Generate GUI
    master = tk.Tk()
    master.title('Pair Trading Backtesting')
    m = tk.IntVar()
    #Text box to enter stock tickers
    tk.Label(master, text="Stock Tickers:").grid(row=0,column=0)
    e1 = tk.Entry(master)
    e1.insert(0, "AAPL")
    e1.grid(row=1,column=0)
    e2 = tk.Entry(master)
    e2.insert(0, "GOOG")
    e2.grid(row=2,column=0)
    #Sliders to set Trade entry/exit parameters
    w1 = tk.Scale(master, label='Long Entry',from_=0, to=-3,digits=3,resolution=0.25, length=200, tickinterval=3)
    w1.set(-1.50)
    w1.grid(row=3,column=0,padx=0, pady=10)
    w2 = tk.Scale(master, label='Short Entry', from_=0, to=3,digits=3,resolution=0.25, length=200,tickinterval=3)
    w2.set(1.50)
    w2.grid(row=3,column=1,padx=0, pady=10)
    w3 = tk.Scale(master, label='Long Exit', from_=1, to=-1,digits=3,resolution=0.25, length=200, tickinterval=2)
    w3.set(0)
    w3.grid(row=3,column=2,padx=0, pady=10)
    w4 = tk.Scale(master, label='Short Exit', from_=-1, to=1,digits=3,resolution=0.25, length=200,tickinterval=2)
    w4.set(0)
    w4.grid(row=3,column=3,padx=0, pady=10)
    w5 = tk.Scale(master, label='Stop Loss',from_=0, to=3,digits=3,resolution=0.25, length=200, tickinterval=3)
    w5.set(1.50)
    w5.grid(row=3,column=4,padx=0, pady=10)
    w6 = tk.Scale(master, label='Window Size', from_=0, to=100,digits=3,resolution=1, length=200,tickinterval=100)
    w6.set(50)
    w6.grid(row=3,column=5,padx=0, pady=10)
    w7 = tk.Scale(master, label='Max Trade Duration', from_=0, to=100,digits=3,resolution=1, length=200,tickinterval=100)
    w7.set(50)
    w7.grid(row=3,column=6,padx=0, pady=10)
    #Button to select regression method
    tk.Label(master, text="Method:").grid(row=0,column=1)
    tk.Radiobutton(master,text="Kalman",padx = 20,variable=m,value=0).grid(row=1,column=1)
    tk.Radiobutton(master,text="OLS",padx = 20,variable=m,value=1).grid(row=2,column=1)
    #Text box to enter strategy duration
    tk.Label(master, text="Strategy Duration:").grid(row=0,column=2)
    f1 = tk.Frame(master)
    e3 = tk.Entry(f1,width=4)
    e3.insert(0, "1")
    e3.pack(side=tk.LEFT)
    tk.Label(f1, text="years").pack(side=tk.LEFT)
    f1.grid(row=1,column=2)
    #Text box to enter initial equity
    tk.Label(master, text="Initial Equity:").grid(row=0,column=3)
    f2 = tk.Frame(master)
    tk.Label(f2, text="$").pack(side=tk.LEFT)
    e4 = tk.Entry(f2,width=10)
    e4.insert(0, "100000")
    e4.pack(side=tk.LEFT)
    f2.grid(row=1,column=3)
    #Text box to enter trade size
    tk.Label(master, text="Trade Size:").grid(row=0,column=4)
    e5 = tk.Entry(master,width=10)
    e5.insert(0, "1000")
    e5.grid(row=1,column=4)
    #Button to perform back test or optimise parameters
    tk.Button(master, text='Trade', command=PerformBacktest).grid(row=1,column=5)
    tk.Button(master, text='Optimise', command=PerformOptimisation).grid(row=1,column=6)
    
    master.mainloop()
    