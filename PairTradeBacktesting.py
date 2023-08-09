# -*- coding: utf-8 -*-
"""
Author: Andrew Killeen
email: killeenandy@gmail.com
Last edited: 09/08/2023

Script to perform pairs trading strategy, using either a rolling OLS or Kalman 
filter to calulate spread, hedge ratio and entry/exit values
"""

import yfinance as yf
import datetime as dt
import numpy as np
import statsmodels.api as sm
from pykalman import KalmanFilter

def TradingStrategy(method,parameters,init_equity,order_qty,pair_data_total,strat_duration):
    """
    Execute strategy between a pair of stocks with desired parameters, Kalman method
    is much faster than OLS so it is recommended to use this method if optimising parameters

    Parameters
    ----------
    method : method for calculating pairwise relationship
    parameters : trade strategy parameters
    init_equity : intial equity available for trading
    order_qty : size of long/short orders
    pair_data_total : pair price data
    strat_duration : duration of strategy
        
    Returns
    -------
    cum_ret : Strategy cumulative returns through time
    draw_down : Strategy draw down through time
    slope : Strategy regression coefficient / hedge ratio through time
    overall_metrics : List of important strategy metrics
    trade_metrics : List of trading metrics
    
    """
    #Parse input parameters, all entry and exit values are in terms of multiples
    #of the spread standard deviation
    long_entry = parameters[0]       #Entry when going long on the spread
    short_entry = parameters[1]      #Entry when going short on the spread
    long_exit = parameters[2]        #Exit when going long on the spread
    short_exit = parameters[3]       #Exit when going short on the spread
    stop_loss = parameters[4]        #Exit when spread goes to large in wrong direction
    window_size = int(parameters[5]) #Size of window for OLS reg, size of burn in period for Kalman
    max_duration = int(parameters[6])#Maximum trad duration in days
    
    #Limit data to desired strategy duration (+ burn in period)
    interval = dt.timedelta(seconds=int(strat_duration*(365.25*24*60*60)))
    start_date = dt.date.today() - interval - dt.timedelta(days=window_size)
    pair_data = pair_data_total[pair_data_total.index >= start_date]
    
    #Initialise trade metrics
    num_days = len(pair_data)        #Number of trading days in strategy
    cum_ret = np.zeros((num_days))   #Cumulative return
    draw_down = np.zeros((num_days)) #Daily drawdown %
    slope = np.zeros((num_days))     #Regression slope / hedge ratio
    intercept = np.zeros((num_days)) #Regression intercept
    
    #Initialise parameters
    portfolio = [0,0,init_equity] #First 2 elements are the amount of each stock held
    #3rd is the additional cash available
    cash = init_equity
    position = None
    trades = []
    
    #Loop through days in strategy 
    for i in range(len(pair_data)):
        #Check burn in period has passed
        if i < window_size:
            continue
        #Calculate hedge quantity and entry/exit conditions
        if method == 'ols':
            fit, curr_std, curr_spread = RollingOLSFit(pair_data,window_size,i)
            
            if position == None:
                curr_hedge = fit.params[1]
            slope[i] = fit.params[1]
            intercept[i] = fit.params[0]      
        elif method == 'kalman':
            if i == window_size:
                slope[:i+1],intercept[:i+1],kf = InitialiseKalman(pair_data.iloc[:i+1,:])
            else:
                kf = UpdateKalman(kf, pair_data.iloc[i,:])
            curr_std = np.sqrt(kf['Q'])
            curr_spread = kf['e']
            
            if position == None:
                curr_hedge = kf['th'][0][0]
            slope[i] = kf['th'][0]
            intercept[i] = kf['th'][1]
            
        #Get current day price data
        price = pair_data.iloc[i,:]
        
        #Check if currently in a trade
        if position == None:
            #Check if either entry condition has been met, if so then perform 
            #trade an update portfolio and position, initialise trade duration
            if curr_spread < long_entry*curr_std:
                entry_cash = cash #To track if a trade was profit making
                trade_value = order_qty*price[1] - np.floor(order_qty*curr_hedge)*price[0]
                cash += trade_value
                portfolio = [np.floor(order_qty*curr_hedge),-order_qty,cash]
                duration = 0
                position = 'long'
            elif curr_spread > short_entry*curr_std:
                entry_cash = cash #To track if a trade was profit making
                trade_value = np.floor(order_qty*curr_hedge)*price[0] - order_qty*price[1]
                cash += trade_value
                portfolio = [-np.floor(order_qty*curr_hedge),order_qty,cash]
                duration = 0
                position = 'short'
        else:
            duration += 1 #Update time spent in trade
            #Check if any exit condition has been met, if so then exit 
            #trade an update portfolio and position
            if position == 'long':
                exit_cond = curr_spread > long_exit*curr_std
                stop_loss_cond = curr_spread < (long_entry - stop_loss)*curr_std 
                time_cond = duration > max_duration
                if (exit_cond or stop_loss_cond or time_cond):
                    trade_value = np.floor(order_qty*curr_hedge)*price[0] - order_qty*price[1]
                    cash += trade_value
                    portfolio = [0,0,cash]
                    position = None
                    #Track trade length and trade return
                    trades += [[duration,((cash-entry_cash)/cash)]] 
            elif position == 'short':
                exit_cond = curr_spread < short_exit*curr_std
                stop_loss_cond = curr_spread > (short_entry + stop_loss)*curr_std  
                time_cond = duration > max_duration
                if (exit_cond or stop_loss_cond or time_cond):
                    trade_value = order_qty*price[1] - np.floor(order_qty*curr_hedge)*price[0]
                    cash += trade_value
                    portfolio = [0,0,cash]
                    position = None
                    #Track trade length and trade return
                    trades += [[duration,((cash-entry_cash)/cash)]]
                    
        #Calculate portfolio value
        portfolio_value = portfolio[0]*price[0] + portfolio[1]*price[1] + portfolio[2]
        if portfolio_value <= 0:
            print('You are now bankrupt! Change your strategy!')
            
        #Calculate returns
        cum_ret[i],draw_down[i] = CalculateReturns(portfolio,portfolio_value,init_equity,cum_ret)
            
    #Estimate risk free rate of return for sharpe and sortino calculation
    rf_ret = FindRiskFreeRate(start_date)
    #Calculate broad strategy metrics
    overall_metrics = FindStrategyMetrics(window_size,cum_ret,draw_down,\
                      trades,strat_duration,init_equity,rf_ret)
    #Calculate trade metrics
    trade_metrics = FindTradeMetrics(trades)                  
    
    return cum_ret,draw_down,slope,overall_metrics,trade_metrics

def RollingOLSFit(pair_data,window_size,i):
    """
    Perform OLS regression of two stocks using rolling window. Use this calculate 
    spread of two stocks and current error in prediction

    Parameters
    ----------
    pair_data : price data for stocks
    window_size : size of regression window
    i : current day of the strategy
        
    Returns
    -------
    fit : regression coefficients
    curr_std : current standard deviation of the spread
    curr_spread : current error in prediction/spread
    
    """
    fit  = sm.OLS(pair_data.iloc[i-window_size:i,0],sm.add_constant(pair_data.iloc[i-window_size:i,1])).fit()
    spread = pair_data.iloc[i-window_size:i,0]-(fit.params[1]*pair_data.iloc[i-window_size:i,1] + fit.params[0])
    curr_std = spread.std()
    curr_spread = pair_data.iloc[i,0]-(fit.params[1]*pair_data.iloc[i,1] + fit.params[0])
    
    return fit, curr_std, curr_spread

def InitialiseKalman(pair_data):
    """
    Initialise Kalman filter for regression coefficients, please see
    py-kalman documentation for full description

    Parameters
    ----------
    pair_data : Price data for stocks
        
    Returns
    -------
    slopes : Estimate of regression slope during burn in period
    intercepts : Estimate of regression intercept during burn in period
    kf : Initial kalman filter parameters
    
    """
    #Initialise params
    kf = {}
    kf['y'] = 0                 #Current observable (price of pair_data[0])
    kf['th'] = np.zeros(2)      #Regression coefficients
    kf['F'] = np.zeros(2)       #Observation matrix
    kf['w'] = 1e-5*np.eye(2)    #Measurement noise
    kf['v'] = 1e-3              #System Noise
    kf['R'] = np.zeros((2,2))   #Transition matrix
    kf['e'] = 0                 #Current error/spread in prediction
    kf['f'] = 0                 #Current mean prediction value
    kf['Q'] = 0                 #Current standard deviation of prediction value
    kf['A'] = 0                 #Transition matrix
    kf['C'] = np.ones((2,2))    #Variance-Covariance matrix for regression coef distriution
    
    #Values of pair_data[1] to used to update filter values
    obs_mat = np.vstack([pair_data.iloc[:,1], np.ones(len(pair_data))]).T[:, np.newaxis]
    #Initialise filter
    initial_kf = KalmanFilter(
        n_dim_obs=1, 
        n_dim_state=2,
        initial_state_mean=kf['th'],
        initial_state_covariance=kf['C'],
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=kf['v'],
        transition_covariance=kf['w']
    )
    #Update relevent parameters based on pair_data[0] data
    th_means, th_covs = initial_kf.filter(pair_data.iloc[:,0])
    kf['th'] = th_means[-1,:][:,np.newaxis]
    kf['C'] = th_covs[-1,:,:]
    
    slopes = th_means[:,0]
    intercepts = th_means[:,1]
    return slopes,intercepts,kf

def UpdateKalman(kf,curr_price):
    """
    Update best estimate of regression coefficients (and other filter params)
    based on todays new price info. Please see InitialiseKalman for description
    of what each parameter is.

    Parameters
    ----------
    kf : Current Kalman filter parameters
    curr_price : current prices of each stock
        
    Returns
    -------
    kf : Updated Kalman filter parameters
    
    """
    kf['F']= np.array([[curr_price[1],1.0]])
    kf['y'] = curr_price[0]
    kf['R'] = kf['C'] + kf['w']
    kf['f'] = kf['F'].dot(kf['th'])
    kf['e'] = kf['y'] - kf['f']
    kf['Q'] = kf['F'].dot(kf['R']).dot(kf['F'].T) + kf['v']
    kf['A'] = kf['R'].dot(kf['F'].T) / kf['Q']
    kf['th'] += kf['A']*kf['e']
    kf['C'] = kf['R'] - kf['A']*kf['F'].dot(kf['R'])
    return kf

def CalculateReturns(portfolio,portfolio_value,init_equity,cum_ret):
    """
    Calculate portfolio returns data
    
    Parameters
    ----------
    portfolio : Current Kalman filter parameters
    portfolio_value : current portfolio value
    init_equity : initial equity for strategy
    cum_ret : daly cumulative returns
        
    Returns
    -------
    curr_cum_ret : cumulative returns for today 
    draw_down : todays draw down
    
    """    
    #From portfolio value calculate cumulative, daily return and drawdown
    curr_cum_ret = (portfolio_value-init_equity)/init_equity
    peak = max((cum_ret+1)*init_equity)
    if portfolio_value >= peak:
        draw_down = 0
    else:
        draw_down = (peak - portfolio_value)/peak
        
    return curr_cum_ret,draw_down

def FindRiskFreeRate(start_date):
    """
    Calculate waht the risk free daily return of strategy would be. This is calculated 
    by finding the annual yield of 10 year treasury bonds at the start date of the
    strategy and estimating the daily return from this.
    
    Parameters
    ----------
    start_date : Strategy start date
        
    Returns
    -------
    rf_ret : Daily risk free return rate
    
    """
    #Find 10 year T-Bill yields
    T10yr = yf.Ticker('^TNX')
    T10yr_hist = T10yr.history(start=start_date,end=start_date+dt.timedelta(days=7))
    #Approximate daily return from annual return / 365
    rf_ret = (0.01*T10yr_hist['Close'].iloc[0])/365
    
    return rf_ret

def FindStrategyMetrics(window_size,cum_ret,draw_down,trades,\
                        strat_duration,init_equity,rf_ret):
    """
    Calculate key strategy metrics

    Parameters
    ----------
    window_size : length of burn in period
    cum_ret : Cumulative daily returns
    daily_ret : Daily returns
    draw_down : Drawdown from peak
    trades : Data on performance of each trade
    strat_duration : duration of strategy
    init_equity : intial equity available for trading
    rf_ret : Daily risk free return rate

    Returns
    -------
    overall_metrics : List of strategy metrics

    """
    #Calculate sharpe ratio, sortino ratio and compund annual growt rate (CAGR)
    portfolio_value = init_equity*(1+cum_ret[-1])
    daily_ret = cum_ret[1:]-cum_ret[:-1]
    sharpe = np.sqrt(len(daily_ret[window_size:]))*(daily_ret[window_size:].mean() - rf_ret)\
        / daily_ret[window_size:].std()
    sortino = np.sqrt(len(daily_ret[window_size:]))*(daily_ret[window_size:].mean() - rf_ret)\
        / daily_ret[window_size:][daily_ret[window_size:]<0].std()
    CAGR = (portfolio_value/init_equity)**(1/strat_duration)-1
    
    overall_metrics = [['Total Return', cum_ret[-1]],\
                       ['Excess Return',cum_ret[-1]-365*rf_ret],\
                       ['Volatility',cum_ret.std()],\
                       ['Sharpe',sharpe],\
                       ['Sortino',sortino],\
                       ['CAGR',CAGR],\
                       ['Max Drawdown', max(draw_down)],\
                       ['Number of Trades', len(trades)]] 
    
    return overall_metrics

def FindTradeMetrics(trades):
    """
    Calculate some metrics of average trade performance

    Parameters
    ----------
    trades : Data on performance of each trade

    Returns
    -------
    trade_metrics : List of metrics summarising trade performance

    """              
    trades = np.array(trades)
    wins = trades[trades[:,1]>0,1]
    losses = trades[trades[:,1]<0,1]
    trade_metrics = [['Ave. Trade Return', trades[:,1].mean()],\
                     ['Winning Trade %', 100*(len(wins)/len(trades))],\
                     ['Ave. Win Value', wins.mean()],\
                     ['Ave. Loss Value', losses.mean()],\
                     ['Max Win Value', max(wins)],\
                     ['Max Loss Value', min(losses)]]
        
    return trade_metrics
