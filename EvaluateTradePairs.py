# -*- coding: utf-8 -*-
"""
Author: Andrew Killeen
email: killeenandy@gmail.com
Last edited: 09/08/2023

Script to assess which pairs of stocks are good candidates for pair trading. 

As the script looks at every combination of stock pairs, if the parameter constraints
are loose this can slow down the script. It is therefore recommended to use tight 
constraints on acceptable parameter values (e.g. v. high correlation, v. low hurst
exponenent)

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import datetime as dt
from tabulate import tabulate

def EvaluateTradePairs(parameters,stock_data):
    """
    Either downloads or updates 'stock_data' - the Closing price info for all
    component stocks. 
    
    Due to delisting of / changes to Russell 2000 components, some of the acquired
    tickers will fail to download data

    Parameters
    ----------
    parameters : List of constraints on acceptable values for pairwise relationship
    parameters
    stock_data : price data for all stocks and indices

    Returns
    -------
    pair_metrics : Dataframe of all stock pairs that fit constraints along with 
    metrics detailing relationship
    
    """
    #Parse input parameters
    period = parameters[0]      #Time period over which to analyse pairs
    corr_lo = parameters[1]     #Lower bound on linear correlation between pairs
    corr_method = parameters[2] #Method to determine linear correlation
    beta_range = parameters[3]  #Desired range of regression slope between pairs 
    adf_hi = parameters[4]      #Upper bound on ADF stationarity test p-val
    hurst_hi = parameters[5]    #Upper bound on acceptable hurst exponent
    hlife_range = parameters[6] #Desired range of halflife of mean reversion time
    
    #Limit data to specified time period
    interval = dt.timedelta(seconds=int(period*(365.25*24*60*60)))
    start_date = dt.date.today() - interval
    stock_data = stock_data[stock_data.index >= start_date]
    stock_data = stock_data.dropna(axis=1,how='any')
    
    #Find linear correlation between all pair combinations and remove any with 
    #a correlation betlow corr_lo
    print('\nSearching for appropriate stock pairs')
    potential_pairs = FindHighCorrelationPairs(stock_data,corr_method,corr_lo)
    
    #Loop through high correlation pairs and assess performance of other metrics,
    #removing pairs that don't fit constraints
    pair_data = []
    for pair in potential_pairs:
        beta = FindPairBeta(stock_data[pair])
        if beta < beta_range[0] or beta > beta_range[1]:
            continue
        
        spread,pval = FindPairADFTestPval(stock_data[pair],beta) 
        if pval > adf_hi:
            continue
    
        hurst = FindPairHurstExp(spread)        
        if hurst > hurst_hi:
            continue
          
        halflife = FindPairHalflife(spread)   
        if halflife < hlife_range[0] or halflife > hlife_range[1]:
            continue
        
        #If stock pair pass all constraints, add to list
        pair_corr = stock_data[pair].corr(method=corr_method)
        pair_data = pair_data + \
            [[pair[0],pair[1],pair_corr.iloc[0,1],beta,hurst,halflife,pval]]
    
    metric_names = ['Index','Stock 1','Stock 2','Corr','Beta','Hurst','Halflife','ADF p-val']
    pair_metrics = pd.DataFrame(pair_data,columns=metric_names[1:])
    
    #Sort pairs according to which has 'most stationary' relationship (lowest AF
    #pval), as this is most pertinent to paird trading
    pair_metrics = pair_metrics.sort_values('ADF p-val',ascending=False)
    #Print first few results so user can choose which to backtest 
    print('\nFound '+str(len(pair_metrics))+' that fit requirements, first '+str(min(10,len(pair_metrics)))+' are:')
    print('\n'+tabulate(pair_metrics.iloc[:10,:-1],headers=metric_names[:-1], floatfmt=".3f", numalign="right"))
    
    return pair_metrics

def FindHighCorrelationPairs(stock_data,corr_method,corr_lo):
    """
    Calcuate linear correlation between all cominations of stock pairs and 
    output the high correlation pairs

    Parameters
    ----------
    stock_data : price data for all stocks and indices
    corr_method : method to calculate linear correlation
    corr_lo: Minimum acceptable correlation strength

    Returns
    -------
    potential_pairs : List of high correlation pairs
    
    """
    #Calculate all pairwise correlations
    r = stock_data.corr(method=corr_method)
    #Create list of high correlation pairs
    potential_pairs = []
    for i in range(len(r)-1):
        new_pairs = r.iloc[i,i+1:][r.iloc[i,i+1:]>corr_lo]
        if len(new_pairs)>0:
            for j in range(len(new_pairs)):
                potential_pairs.append([r.columns[i],new_pairs.index[j]])
                
    return potential_pairs

def FindPairBeta(pair_data):
    """
    Calculate the slope of the regression between the two stocks

    Parameters
    ----------
    pair_data : price data for pair stocks

    Returns
    -------
    beta : Lslope of regression betwee stocks
    
    """
    ols_res  = sm.OLS(pair_data.iloc[:,0],sm.add_constant(pair_data.iloc[:,1])).fit()
    beta = ols_res.params[1]
    
    return beta

def FindPairADFTestPval(pair_data,beta):
    """
    Perform augmented Dickey-Fuller test on the spread of the pair data to test
    for stationarity 

    Parameters
    ----------
    pair_data : price data for pair stocks

    Returns
    -------
    spread : The spread between the pair when accounting for the regression
    pval : p-val for the process being stationary
    
    """
    spread = pair_data.iloc[:,0]-beta*pair_data.iloc[:,1]
    spread = spread.to_numpy()
    adf_res = adfuller(spread)
    pval = adf_res[1]
    return spread,pval

def FindPairHurstExp(spread):
    """
    Calculate the hurst exponent for the spread, another measure of stationarity
    Parameters
    
    ----------
    spread : The spread between the pair when accounting for the regression

    Returns
    -------
    hurst : Hurst exponent value
    
    """
    lags = range(2, 20)
    tau = [np.sqrt(np.std(spread[lag:]-spread[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0]*2.0
    
    return hurst

def FindPairHalflife(spread):
    """
    Calculate the half life, in days, for the time for the spread to revert back
    to its mean value
    
    ----------
    spread : The spread between the pair when accounting for the regression

    Returns
    -------
    halflife : Half life of mean reversion time in days
    
    """
    spread_lag = np.roll(spread,1)
    spread_lag[0] = spread_lag[1]
    spread_ret = spread - spread_lag
    spread_ret[0] = spread_ret[1]
    spread_lag2 = sm.add_constant(spread_lag)
    hlife_res = sm.OLS(spread_ret,spread_lag2).fit()
    halflife = -np.log(2) / hlife_res.params[1]
    
    return halflife

if __name__ == "__main__":
    period = 2
    corr_lo = 0.95
    corr_method = 'pearson'
    beta_range = [0.5,2]
    adf_hi = 0.001
    hurst_hi = 0.4
    hlife_range = [3,15]
    
    parameters = [period,corr_lo,corr_method,beta_range,adf_hi,hurst_hi,hlife_range]
    stock_data = pd.read_pickle("stock_data.pkl")
    pair_metrics = EvaluateTradePairs(parameters,stock_data)

    
    
    