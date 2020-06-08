# Making a function StockVol to calibrate stock volitility under geometric Brownian motion model

import numpy as np
import matplotlib as m
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np
import datetime as dt
from datetime import date
yf.pdr_override()

#USE    : To get historical data for a particular stock
#INPUT  : code of the company (ticker: GOOG)
#OUTPUT : array of close price of data
def GetData(companyName, years=1):
    # Getting today's date
    today = date.today()
    # Setting start date to historical years
    startDate = today.replace(today.year-years)
    # Date : Open High Low Close "Adj Close" Volume
    data = pdr.get_data_yahoo(companyName, startDate, today)
    # Cleaning to get the close prices only
    closePrice = [data["Close"][i] for i in range(len(data))]
    # Return close price of the data
    return closePrice

#USE        : to find the volitility of the stock data
#INPUT      : array of 1 year historical prices
#ASSUMPTION : Stock doesn't pay dividends
#OUTPUT     : historical volitility of the stock (standard deviation)
def StockVol(histoPrice):
    #Finding the ratio of current/previous values
    Sn = [np.log(histoPrice[i+1]/histoPrice[i]) for i in range(len(histoPrice)-1)]
    return np.sqrt(np.var(Sn, ddof=1))

#USE        : to generate n stock path
#INPUT      : n-> number of stock path, sigma-> volitility, 
#           : T-> TerminalTime (yearly unit), nT-> numberOfTimePeriods
#           : r-> interest rate, delta-> continuous dividend yield 
#           : S0-> initial price of the stock
#ASSUMPTION : Stock is not paying any dividends
#OUTPUT     : Stock Path as a matrix
def StockPath(n, sigma = 0, S0=0, T=1, nT=10, r=.02, delta=0):
    path = []
    for period in range(nT):
        # Step size for every path simulation
        step =  T/n
        #
        periodPrice = [S0]
        # Computation of n simulated stock prices 
        # np.random.normal(0,1,n) outputs array of n normal values
        Y = np.exp(((r-(sigma**2)/2)*step)+(sigma*np.random.normal(0,1,n)*np.sqrt(step)))
        # now we have array of simulated factor for n stock path 
        # we want to multiply this to previous to find simulated price
        counter = 0
        for y in Y:
            periodPrice.append(periodPrice[counter]*y)
            counter += 1
        path.append(periodPrice)
    return(path)
            
        
    









