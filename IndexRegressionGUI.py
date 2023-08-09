# -*- coding: utf-8 -*-
"""
Author: Andrew Killeen
email: killeenandy@gmail.com
Last edited: 09/08/2023

Script to perform regression of a chosen stock index onto a selection of chosen
stocks. These are input, along with the desired time period over which to evaluate the 
relationship, in a GUI. The fitted regression coefficients are then output, along
with the performance of the model using test data
"""

import tkinter as tk
from DataLoader import LoadData
from tabulate import tabulate
import statsmodels.api as sm
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

def ParseRegression():
    """
    Parse input from the GUI, perform regression and output results
    """
    if m.get() == 0:
        index = '^SPX'
    elif m.get() == 1:
        index = '^RUT'
    elif m.get() == 2:
        index = '^NDX'
    index_data = stock_data[index]
    
    period = int(e1.get())
    
    tickers = []
    for ent in entries:
        if len(ent.get()) == 0:
            continue
        tickers.append(ent.get())
    
    chosen_data = stock_data[tickers]
    cleaned_data = chosen_data.dropna(axis=1,thresh=5)
    if len(chosen_data.columns)!=len(cleaned_data.columns):
        lost_stocks = set(chosen_data.columns)^set(cleaned_data.columns)
        print('Do not have sufficient data for all stocks, please replace: '+\
              str(lost_stocks))
        return None
    
    metrics = PerformRegression(index_data,period,chosen_data)

    print('\nRegression Coefficients:')
    print('\n'+tabulate(metrics[0],headers=['Variable','Value','Std. Err.'], \
                        floatfmt=".3f", numalign="right"))

    print('\nRegression Metrics:')
    print('\n'+tabulate(metrics[1],headers=['Metric','Value'], \
                        floatfmt=".5f", numalign="right"))

    return

def PerformRegression(index_data,period,stock_data):  
    """
    Perform analysis over desired time period and evaluate performance

    Parameters
    ----------
    index_data : price data for chosen index
    period : time period over which to perform regression
    stock_data : price data for chosen stocks
        
    Returns
    -------
    metrics : Regression coefficients and performance metrics
    
    """
    #Find start date for anaylsis
    interval = dt.timedelta(seconds=int(period*(365.25*24*60*60)))
    start_date = dt.date.today() - interval 
    #Restrict data to requested time period
    stock_data = stock_data[stock_data.index >= start_date]
    index_data = index_data[index_data.index >= start_date]
    #Convert price data to daily returns data
    stock_cum_ret = stock_data.diff()/stock_data
    index_cum_ret = index_data.diff()/index_data 
    #Fit model and test performance
    results,train_RMSE,test_RMSE,test_R2 = FitAndTestModel(stock_cum_ret,index_cum_ret)
    
    #Aggregate cofficient and standard error values
    reg_results = []
    for i in range(len(results.params)):
        reg_results += [[results.params.index[i],results.params[i],results.bse[i]]]
    #Aggregate test and training performance metrics
    reg_perf = [['Training R-Squared',results.rsquared],\
                ['Training RMSE',train_RMSE],\
                ['Test R-Squared',test_R2],\
                ['Test RMSE',test_RMSE]]
        
    metrics = [reg_results,reg_perf]
    
    return metrics

def FitAndTestModel(stock_ret,index_ret):
    """
    Fit regression model

    Parameters
    ----------
    stock_ret : stock daily ret
    index_ret : index daily ret
        
    Returns
    -------
    results : Regression results
    train_RMSE : training data root mean squared error
    test_RMSE : test data root mean squared error
    test_R2 : test data R-squared value
    
    """
    X = sm.add_constant(stock_ret)
    Y = index_ret
    X_train, X_test, Y_train, Y_test = train_test_split(X.iloc[1:,:],Y.iloc[1:])
    results = sm.GLS(Y_train, X_train, missing='drop').fit()
    res_arr = results.params.to_numpy()[np.newaxis,:]
    Y_train_pred = np.squeeze(res_arr.dot(X_train.T))
    train_RMSE = np.sqrt(np.mean((Y_train_pred-Y_train)**2))
    
    Y_test_pred = np.squeeze(res_arr.dot(X_test.T))
    test_RMSE = np.sqrt(np.mean((Y_test_pred-Y_test)**2))
    test_R2 = r2_score(Y_test,Y_test_pred)
    
    return results,train_RMSE,test_RMSE,test_R2


if __name__ == '__main__':
    #Get stock data
    update_data = False
    
    
    if update_data:
        stock_data = LoadData(redownload_all = False, years = 10)
    else:
        stock_data = pd.read_pickle("stock_data.pkl")
    #Generate GUI 
    master = tk.Tk()
    master.title('Index Regression')
    m = tk.IntVar()
    #Add in button to select index
    tk.Label(master, text="Index:").pack()
    tk.Radiobutton(master,text="S&P 500",pady = 5,padx = 20,variable=m,value=0).pack()
    tk.Radiobutton(master,text="Russell 2000",pady = 5,padx = 20,variable=m,value=1).pack()
    tk.Radiobutton(master,text="Nasdaq 100",pady = 5,padx = 20,variable=m,value=2).pack()
    #Add in text box to enter time period to analyse
    tk.Label(master, text="Time Period:").pack(pady = 5)
    f1 = tk.Frame(master)
    e1 = tk.Entry(f1,width=3)
    e1.pack(side=tk.LEFT)
    tk.Label(f1, text="years").pack(side=tk.LEFT)
    f1.pack(pady = 5)
    #Add in textboxes to input stock tickers for regression
    tk.Label(master, text="Stock Tickers:").pack(pady = 5)
    entries = []
    init_suggestions = ['AAPL','GOOG','META','MSFT','TSLA']
    for i in range(10):
        row = tk.Frame(master)
        lab = tk.Label(row, width=5, text=str(i+1), anchor='w')
        ent = tk.Entry(row)
        if i < len(init_suggestions): 
            ent.insert(0, init_suggestions[i])
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((ent))
    #Add button to perform the analysis
    b1 = tk.Button(master, text='Analyse',command=ParseRegression)
    b1.pack(padx=5, pady=5)
    
    master.mainloop()
