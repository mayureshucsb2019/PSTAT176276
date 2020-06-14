#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
# !pip3 install pandas_datareader
# !pip3 install yfinance
# !pip3 install cvxpy
import numpy as np
import matplotlib as m
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data
import yfinance as yf
import datetime as dt
from datetime import date
import scipy.stats as stats
import cvxpy as cp
yf.pdr_override()


# In[2]:


#USE    : To get historical data for a particular stock
#INPUT  : ticker of the company
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

#USE    : To get continuous annual dividend yield for a stock
#INPUT  : ticker of the company (as a string), number of dividend payments in a year
#OUTPUT : continuous annual dividend yield
def ContDiv(companyName):
    # Get stock info
    stock = yf.Ticker(companyName)
    # Get current stock price
    S0 = pdr.get_data_yahoo(companyName).iloc[-1,:]['Close']
    if len(stock.dividends) > 0:
        # get frequency of dividend payments in a year
        div_df = stock.actions[['Dividends']]
        div_df = div_df[div_df['Dividends'] > 0]
        freq = len(div_df['2019'])
        # compute annual dividend yield
        annualDY = (stock.dividends[-1]*freq)/S0
    else:
        annualDY = 0
    # converts annual dividend yield to effective continuous yield
    delta = np.log(annualDY + 1)  
    return delta

#USE        : to find the historical volitility of the stock
#INPUT      : array of 1 year historical prices
#ASSUMPTION : Stock doesn't pay dividends
#OUTPUT     : historical volitility of the stock (standard deviation)
def StockVol(histoPrice):
    #Finding the ratio of current/previous values
    Sn = [np.log(histoPrice[i+1]/histoPrice[i]) for i in range(len(histoPrice)-1)]
    return np.sqrt(np.var(Sn, ddof=1))

#USE        : to generate n stock path
#INPUT      : nSteps-> number of time steps, sigma-> volatility, 
#           : T-> TerminalTime (yearly unit), nSimulations-> number of Simulations
#           : r-> interest rate, delta-> continuous dividend yield 
#           : S0-> initial price of the stock
#ASSUMPTION : Stock is not paying any dividends
#OUTPUT     : Stock Path as a matrix
def StockPath(nSteps, sigma, S0, T, nSimulations, r, delta):
    path = []
    for period in range(nSimulations):
        # Step size for every path simulation
        step =  T/nSteps
        #
        periodPrice = [S0]
        # Computation of n simulated stock prices 
        # np.random.normal(0,1,n) outputs array of nSteps normal values
        Y = np.exp(((r-delta-(sigma**2)/2)*step)+(sigma*np.random.normal(0,1,nSteps)*np.sqrt(step)))
        # now we have array of simulated factor for nSteps stock path 
        # we want to multiply this to previous to find simulated price
        counter = 0
        for y in Y:
            periodPrice.append(periodPrice[counter]*y)
            counter += 1
        path.append(periodPrice)
    return(path)      

#USE        : To generate the European put option price through Monte Carlo method
#INPUT      : StockPath, nSteps, 
#           : T-> TerminalTime (yearly unit),
#           : r-> interest rate, delta-> continuous dividend yield,
#           : S0-> initial price of the stock, K-> Strike Price
#OUTPUTS    : Discounted Payoff Vector, Euro Put Option Price, Variance of Price
def EurOptPrice(StockPath, T, r, K):
    # number of simulations
    n = np.shape(StockPaths)[0]
    # number of timesteps
    nsteps = np.shape(StockPaths)[1]
    # Last Column of StockPath is Terminal Value
    Dis_Payoff_Vec = []
    for j in range(0,n):
        ST_j = StockPath[j][nsteps-1]
        # Create Column of Disounted Payoffs
        Dis_Payoff_j = np.exp(-r*T)*max(0,K - ST_j)
        Dis_Payoff_Vec = np.append(Dis_Payoff_Vec,Dis_Payoff_j)
        Price = np.mean(Dis_Payoff_Vec)
        Price_Var = np.var(Dis_Payoff_Vec)
    return (Dis_Payoff_Vec, Price, Price_Var)

#### compute true Black-Scholes price of the option
# define function for computing Black-Scholes price for Call or Put
# inputs: SO = initial stock price, K = strike price, r = risk-free interest rate, 
#         T = maturity time, sig = sigma (volatility), type = 'C' for Call or 'P' for Put
def BlackScholes(S0, K, r, T, sigma, optionType):
    if optionType=="C":
        d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
  
        value = S0*stats.norm.cdf(d1,0,1) - K*np.exp(-r*T)*stats.norm.cdf(d2,0,1)
  
    elif optionType=="P":
        d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
  
        value =  (K*np.exp(-r*T)*stats.norm.cdf(-d2, 0, 1) - S0*stats.norm.cdf(-d1,0,1))
    
    return(value)


#USE        : To generate the Aemrican put option price through Monte Carlo method
#INPUT      : StockPaths, nSteps-> number of time steps, sigma-> volatility, 
#           : T-> TerminalTime (yearly unit), nSimulations-> number of Simulations
#           : r-> interest rate, delta-> continuous dividend yield
#           : S0-> initial price of the stock, K-> Strike Price
#ASSUMPTION : Stock is not paying any dividends
#OUTPUTS    : Discounted Payoff Vector, American Put Option Price, Variance of the Price

# number of MC samples
mcSamples = 1000

def AmericanOptionPrice(StockPaths, sigma, r, K, T, mcSamples, delta):
    # number of simulations
    nSimulations = np.shape(StockPaths)[0]
    # number of timesteps
    nSteps = np.shape(StockPaths)[1] - 1
    # step size
    step = T/nSteps
    # Shape is (nSimulations, nSteps)
    # Take a closer look, all stock path has been reversed to work in the backward direction
    # i.e. it starts from time in reverse direction
    paths = np.array(StockPaths).T[::-1] 
    #print(f"shape of paths is {np.shape(paths)}")
    
    # This contains payoffs of all the path or 0th time step
    exercisedPayoffs = [(K-paths[0]).clip(min=0)]
    exercisedExpDisPayoffs = [(K-paths[0]).clip(min=0)]
    
    for i in range(1,nSteps):
        # Has number of values equal to nSsimulations i.e. every path value
        # So basically variable 'paths' contain price at each time step for all simulation
        # stockPrices has price at time step i for every path of simulation
        stockPrices = paths[i]
        #print(f"stockPrices shape is {np.shape(stockPrices)}")
        
        # Make positive payoff using clip, this is payofsf at time step i for all path 
        payOffs = (K - stockPrices).clip(min=0)
        #print(payOffs, np.shape(payOffs))
        
        # We need to find Discounted Conditional Expected Price for next step of each step value
        # So we need to Simulate 1 step Monte Carlo to generate next time stock price
        # Find the payoff of this next time stock price and calculate average 
        # We use blackSholes model for price simulation
        
        # For every path in the stockPrices simulate Monte Carlo and get next prices using Black-Sholes
        # time step is ith
        simulatedPrices = [ stockPrice*np.exp(sigma*np.random.normal(0,1,mcSamples)*np.sqrt(step) 
                                              + (r-(sigma**2)/2)*step) for stockPrice in stockPrices]
        simulatedPrices = np.array(simulatedPrices)
        #print(f"Shape of simulatedPrices is: {np.shape(simulatedPrices)}")
        # shape is nSimulations X mcSamples
        # So for every path at timeStep i we have mcSamples montecarlo samples simulated
        
        #If this is the t-1 step then the expected payoff is to be simulated mean
        if(i == 1):
            # Find expected payoff by taking mean of all the path
            expectedPayoffs = np.array([np.mean(expectedPayoff) 
                                        for expectedPayoff in K-simulatedPrices.clip(min=0)])
            #print(f"Shape of expectedPayoffs is {np.shape(expectedPayoffs)}")
            discountedExpectedPayoffs = expectedPayoffs* np.exp(-r*step)
            #print(f"Shape of discountedExpectedPayoff is {np.shape(discountedExpectedPayoffs)}")
            
        # For other time step it must be done by machine learning
        else:
            # discountedExpectedPayoffs needs to be calculated using the machine learning
            # Take as input stockPrices (has stock price for each path at time step i)
            # and take another input as discountedExpectedPayoffs (has expectedPayoffs 
            # for each path at time step i)
            # run the machine learning model on this to get the weights
            # use this weights to predict new expected payoff at timestep i-1
            
            X = np.array([np.array([1]*len(stockPrices)), stockPrices, stockPrices**2])
            X = X.T
            #print(np.shape(X))
            #print(X)
            # Formalizing the regression model
            beta = cp.Variable(3)
            loss = cp.sum_squares(discountedExpectedPayoffs-X@beta)
            prob = cp.Problem(cp.Minimize(loss))
            prob.solve()
            # estimating discounted expected payoffs using estimated parameters
            discountedExpectedPayoffs = X@beta.value
        exercisedPayoffs.insert(0, payOffs)
        exercisedExpDisPayoffs.insert(0, discountedExpectedPayoffs)
    exercisedPayoffs = np.array(exercisedPayoffs).T
    exercisedExpDisPayoffs = np.array(exercisedExpDisPayoffs).T
    #Now we have nSimulations X nSteps shaped arrays
    discountedPayoffsMax = []
    for i in range(nSimulations):
        for j in range(nSteps):
            exerPayoff = exercisedPayoffs[i][j]
            exerDisPayoff = exercisedExpDisPayoffs[i][j]
            if exerPayoff > exerDisPayoff :
                disPayoff = exerPayoff*np.exp(-r*step*j)
                break
            else:
                disPayoff = exerDisPayoff*np.exp(-r*step*j)
        discountedPayoffsMax.append(disPayoff) 
        
    return(discountedPayoffsMax, np.mean(discountedPayoffsMax), np.var(discountedPayoffsMax, ddof=1))


# In[3]:


# # import 1 yr LIBOR rates from 6/10/20 to a year back
# LIBOR = pd.read_csv("1-yr-LIBOR-rates.csv")
# # current date
# endDate =  dt.datetime.strptime("2020-06-02", '%Y-%m-%d').date()
# # start date
# startDate = endDate.replace(year=2019, day=3)
# # subset interest rates 
# # LIBOR[(LIBOR['date'] >= startDate) & (LIBOR['date'] <= currentDate)]
# R = LIBOR[str(startDate):str(endDate)]
# print(len(R))


# In[4]:


np.random.seed(1)
# define parameters for Put Option pricing
HistoPrice = GetData("GOOG", years=1)
sigma = StockVol(HistoPrice)
delta = ContDiv("GOOG")
T = 1
r1 = 0.0063  # current 12-month USD LIBOR rate (as of 06/10/20)
r = np.log(r1+1)  # equivalent continous interest rate
K = 1200
nSimulations = 100
S0 = HistoPrice[0]
nSteps = len(HistoPrice)
# generate stock paths
StockPaths = StockPath(nSteps, sigma, 
                       S0, T, nSimulations, r, delta)

# implement Euro Put Option pricing
EuroPayoffs, EuroPrice, EuroVar = EurOptPrice(StockPaths, T, r, K)
print("Length of Euro Payoff Vector: ", len(EuroPayoffs))
print("Euro Put Price: ", EuroPrice)
print("Euro Price Variance: ", EuroVar)

# implement American Option Pricing function
AmerPayoffs, AmerPrice, AmerVar = AmericanOptionPrice(StockPaths, sigma, r, K, T, mcSamples, delta)
print("Length of American Payoff Vector: ", len(AmerPayoffs))
print("American Put Price: ", AmerPrice)
print("American Price Variance: ", AmerVar)


# In[5]:


# control variate method

# control variate function
# USE   : reduce the variance of the simulated data
# INPUTS: y-discounted payoff of American option for every simulated path; 
#         x-discounted payoff of European option for each path; 
#         mu-mean value of x;
# OUTPUT: estimated price of American option after using control variate

def contvariate(x,mu,y):
    c = -np.cov(x,y)[1,0]/np.var(x,ddof=1)
    return(np.mean(y+c*(x-mu)))

# implement control variate method
x = EuroPayoffs
mu_x = BlackScholes(S0, K, r, T, sigma, "P")
y = AmerPayoffs

print("American Put Option Price w/ Control Variate: ", contvariate(x,mu_x,y))

