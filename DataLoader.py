# -*- coding: utf-8 -*-
"""
Author: Andrew Killeen
email: killeenandy@gmail.com
Last edited: 09/08/2023

Script to download end of day price data for S&P 500, Nasdaq 100 and Russell 2000
components from Yahoo finance. Either all data is downloaded for a specified number
of years or the current database is updated with any new data from the last update 
"""

import yfinance as yf
import pandas as pd
import datetime as dt
import tabula
import pandas_market_calendars as mcal

def LoadData(redownload_all = False, years = 10):
    """
    Either downloads or updates 'stock_data' - the Closing price info for all
    component stocks. 
    
    Due to delisting of / changes to Russell 2000 components, some of the acquired
    tickers will fail to download data

    Parameters
    ----------
    redownload_all : Boolean for whether to redownload data for all stocks or 
    just update existing data. The default is False.
    years : If redownload_allis True, this is the number of years of data
    to download. The default is 10.

    Returns
    -------
    stock_data : Closing price data for all indices and their component stocks

    """
    #Obtain start and end dates for data to be downloaded
    if redownload_all:
        max_days = years*365
        interval = dt.timedelta(days=max_days)
        end_date = dt.date.today()
        start_date = end_date - interval 
    else:
        #If updating data, check if there have been any trading days since last
        #update, if not then data is up to date
        stock_data = pd.read_pickle("stock_data.pkl")
        end = dt.date.today() - dt.timedelta(days=1)
        nyse = mcal.get_calendar('NYSE').\
            schedule(start_date=end - dt.timedelta(days=7), end_date=end)
        trading_dates = nyse.index.date
        if trading_dates[-1]==stock_data.index[-1]:
            print('\nStock data is up to date')
            return stock_data
        else:
            print('\nAdding most recent data to stock data')
            end_date = dt.date.today()
            start_date = stock_data.index[-1] + dt.timedelta(days=1)

    #Download performance of 3 indices
    print('\nCommencing download of indices performance')
    index_data = GetIndicesData(start_date,end_date)
    #Download performance of S&P 500 components
    print('\nCommencing download of S&P 500 components performance')
    SNP_data = GetSP500ComponentData(start_date,end_date)
    #Download performance of Russell 2000 components
    print('\nCommencing download of Russell 2000 components performance')
    R2000_data = GetR2000ComponentData(start_date,end_date)
    #Download performance of Nasdaq 100 components
    print('\nCommencing download of Nasdaq 100 components performance')
    N100_data = GetN100ComponentData(start_date,end_date)
    N100_info = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
    N100_data = yf.download(N100_info.Ticker.to_list(),start_date,end_date, auto_adjust=False,progress=False)['Close']
    #Aggregate data and remove components common to multiple indices
    combined_data = CombineData(index_data,SNP_data,R2000_data,N100_data)
    
    if redownload_all:
        stock_data = combined_data
    else:
        stock_data = pd.concat([stock_data,combined_data])
    #Remove any spurious dates
    stock_data=stock_data.dropna(axis=0,thresh=10)
    #Save new data
    stock_data.to_pickle("stock_data.pkl")
    
    return stock_data

def GetIndicesData(start_date,end_date):
    """
    Download data for 3 indices of interest
    
    Parameters
    ----------
    start_date : Starting date from which to download data
    end_date : Ending date from which to download data

    Returns
    -------
    index_data : Closing price data for the 3 indices
    
    """
    index_tickers = ['^SPX','^NDX','^RUT']
    index_data = yf.download(index_tickers,start_date,end_date, auto_adjust=False,progress=False)['Close']
    
    return index_data

def GetSP500ComponentData(start_date,end_date):
    """
    Scrape S&P 500 tickers and use these to required download component data
    
    Parameters
    ----------
    start_date : Starting date from which to download data
    end_date : Ending date from which to download data

    Returns
    -------
    SNP_data : Closing price data for S&P 500 component stocks
    
    """
    #Get tickers from wikipedia page
    SNP_info = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    #Correct 2 incorrect tickers
    SNP_info.loc[65,'Symbol']='BRK-B'
    SNP_info.loc[81,'Symbol']='BF-B'
    SNP_data = yf.download(SNP_info.Symbol.to_list(),start_date,end_date, auto_adjust=False,progress=False)['Close']
    
    return SNP_data

def GetR2000ComponentData(start_date,end_date):
    """
    Scrape Russell 2000 tickers and use these to required download component data
    
    Parameters
    ----------
    start_date : Starting date from which to download data
    end_date : Ending date from which to download data

    Returns
    -------
    R2000_data : Closing price data for Russell 2000 component stocks
    
    """
    R2000_info=[]
    #Read pdf from FTSE Russell website, tickers are on multiple pages so must 
    #loop through to scrape them all
    R2000_pdf=tabula.read_pdf('https://content.ftserussell.com/sites/default/files/ru2000_membershiplist_20220624_0.pdf',stream=True,pages='all')
    for page in R2000_pdf:
        R2000_info += pd.concat([page['Unnamed: 0'].iloc[1:],page['Unnamed: 2'].iloc[1:]]).dropna().to_list()
    R2000_data = yf.download(R2000_info,start_date,end_date, auto_adjust=False,progress=False)['Close']
    
    return R2000_data

def GetN100ComponentData(start_date,end_date):
    """
    Scrape Nasdaq 100 tickers and use these to required download component data
    
    Parameters
    ----------
    start_date : Starting date from which to download data
    end_date : Ending date from which to download data

    Returns
    -------
    N100_data : Closing price data for Nasdaq 100 component stocks
    
    """
    #Scrape tickers from wikipedia
    N100_info = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
    N100_data = yf.download(N100_info.Ticker.to_list(),start_date,end_date, auto_adjust=False,progress=False)['Close']
    
    return N100_data

def CombineData(index_data,SNP_data,R2000_data,N100_data):
    """
    Combine all downloaded data into one dataframe and remove dupilicates of 
    stocks that are in more than one index

    Parameters
    ----------
    index_data : Closing price data for indices
    SNP_data : Closing price data for S&P 500 components
    R2000_data : Closing price data for Russell 2000 components
    N100_data :  Closing price data for Nasdaq 100 components

    Returns
    -------
    combined_data : Combined price data for all stocks and indices

    """
    combined_data = pd.concat([index_data,SNP_data],axis=1)
    combined_data = pd.concat([combined_data,R2000_data],axis=1)
    combined_data = pd.concat([combined_data,N100_data],axis=1)
    #Impute stray nan values
    combined_data.fillna(method='ffill',limit=1)
    #Drop stocks for which there is no data
    combined_data = combined_data.dropna(axis=1,how='all')
    #Remove dupilicates of stocks that are in more than one index
    combined_data = combined_data.loc[:,~combined_data.columns.duplicated()].copy()
    #Remove timestamp from dataframe index so it is a just a date object
    combined_data.index  = pd.to_datetime(combined_data.index).date
    
    return combined_data

if __name__ == "__main__":
    stock_data = LoadData()