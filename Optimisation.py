# -*- coding: utf-8 -*-
"""
Author: Andrew Killeen
email: killeenandy@gmail.com
Last edited: 09/08/2023

Script to optimise trade entry/exit parameters using the DyCors algorithm.

Parameters are optimised to maximise strategy returns.
"""

import numpy as np
from math import log
from PairTradeBacktesting import TradingStrategy

def Optimise(method,init_equity,order_qty,pair_data,strat_duration):
    """
    Perform DyCors optimisation algorithm for strategy parameters. Please see 
    https://www.tandfonline.com/doi/abs/10.1080/0305215X.2012.687731 
    for details of methodology and implementation.
    
    Parameters are optimised to maximise strategy returns.

    Parameters
    ----------
    method : method for calculating pairwise relationship
    init_equity : intial equity available for trading
    order_qty : size of long/short orders
    pair_data_total : pair price data
    strat_duration : duration of strategy
        
    Returns
    -------
    opt_params : optimised parameter values
    cum_ret : Strategy cumulative returns through time
    draw_down : Strategy draw down through time
    slope : Strategy regression coefficient / hedge ratio through time
    overall_metrics : List of important strategy metrics
    trade_metrics : List of trading metrics
    
    """
    
    nV = 7                                      # number of parameters to optimise
    nP = 20                                     # size of intial population
    sig0 = 0.2                                  # initial step size
    sigmin = sig0*(0.5**6)                      # minimum step size
    Nmax = 50                                   # max number of strategies to backtest
    nNei = nV*100                               # number of neighbours
    bL = 0                                      # lower bound
    bU = 1                                      # upper bound
    n0 = nP                                     # Initial population size
    n = n0                                      # Initialise function evals
    sig = sig0                                  # Initialise step size
    C_f = 0                                     # Initialise failure counter
    C_s = 0                                     # Initialise success counter
    weights = [0.3,0.5,0.8,0.95]                # Exploitation weighting
    
    #Generate initial list of nP set of parameter values using a latin hypercube
    xP = LatinHyperCube(nV,n0)
    #Determine performance of initial sets of values
    fP = np.zeros(len(xP))
    for i,x in enumerate(xP):
        x_scale = RescaleInputs(x)
        cum_ret,_,_,_,_ = TradingStrategy(method,x_scale,init_equity,order_qty,\
                                         pair_data,strat_duration)   
        fP[i] = -cum_ret[-1]
    
    #Determine current best set of parameters
    ib = fP.argmin()                         # determine best solution
    xB,fB = xP[ib,:],fP[ib]                     # Select best x and f
    xB = xB[np.newaxis]
    
    while n < Nmax:
        #Build a surrogate surface of parameter space
        s  = GenerateSurrogateRBF(xP,fP.T) 
        x_new = xB     # initialise in nested while loop
        #Generate new parameter set to evaluste strategy on
        while np.sum(x_new-xB) == 0:
            #Calcualte probability of perturbing a variable in xB
            Pselect = MutationProbability(n,n0,Nmax,nV)
            if Pselect == 0:
                Pselect = 0.01
            #Generate nNei parameter sets based off the current best set xB
            y = GenerateNeighbours(xB,nNei,Pselect,sig,bL,bU)
            #Evaluate surrogate model at trial point
            y = y.T
            fy = EvaluateRBF(xP,s,y)
            #Select next eval point
            x_new = SelectNextEvalPoint(xP,y,fy,nNei,n,n0,weights)
    
        #Perform Function evluation
        x_scale = RescaleInputs(x_new)
        cum_ret,_,_,_,_ = TradingStrategy(method,x_scale,init_equity,order_qty,\
                                          pair_data,strat_duration)
        f_new = -cum_ret[-1]
        
        # Update counters and step size
        C_f,C_s,sig = UpdateCountersAndStep(f_new,fB,C_f,C_s,sig,sigmin,nV)

        # Update best solution
        if f_new < fB:
            xB, fB = x_new, f_new
        #Add new evaluation point to list
        x_new = x_new[np.newaxis]
        f_new = f_new[np.newaxis]
        xP = np.concatenate((xP,x_new),axis=0)
        fP = np.concatenate((fP,f_new),axis=0)
        
        if n % 10 == 0:
            print('\nParameters '+str(int(100*n/Nmax))+'% optimised')
            
        n += 1
    
    #Perform final back test on best parameter set and return
    xB = RescaleInputs(xB)
    cum_ret,draw_down,slope,overall_metrics,trade_metrics = \
        TradingStrategy(method,xB,init_equity,order_qty,pair_data,strat_duration)
    
    opt_params = [['Long Entry',xB[0]],\
                  ['Short Entry',xB[1]],\
                  ['Long Exit',xB[2]],\
                  ['Short Exit',xB[3]],\
                  ['Stop Loss',xB[4]],\
                  ['Window Size',xB[5]],\
                  ['Max Trade Duration',xB[6]]]
        
    return opt_params,cum_ret,draw_down,slope,overall_metrics,trade_metrics

def RescaleInputs(x):
    """
    Rescale normalised input parameter to required range (same range as those
                                              listed on the slider in the GUI)
    Parameters
    ----------
    x : list of parameter inputs normalised between 0 and 1
        
    Returns
    -------
    x_temp : parameter inputs scaled to correct values for backtest
    
    """
    x_temp = x.copy()
    x_temp[0] *= -3
    x_temp[1] *= 3
    x_temp[2:4] = -1 + 2*x_temp[2:4]
    x_temp[4] *= 3
    x_temp[5] = 1 + int(99*x_temp[5])
    x_temp[6] = 1 + int(99*x_temp[6])
    return x_temp

def MutationProbability(n,n0,Nmax,d):
    """
    Determine the probability of perturbing a parameter in xB when generating 
    possible new evaluation points. This should decrease as the number of 
    backtests perfromed increases
    
    Parameters
    ----------
    n : current number of backtests performed
    n0 : initial number of backtest performed
    Nmax : maximum number of backtests to be performed
    d : number of parameters
        
    Returns
    -------
    phi : Probability of preturbing a given parameter
        
    """
    p0  = min(20/d,1)
    phi = p0*(1 - log(n-n0+1)/log(Nmax-n0))
    return phi

def Mutate(newP,Pselect,sig,nNei,bL,bU):
    """
    Generate new set of trial parameters
    
    Parameters
    ----------
    newP : Initialised set of trail points
    Pselect : probability of mutating a variable
    sig : step size for mutation
    nNei : number of neighbouring points to generate
    bL : lower bound for normalised parameter
    bU : upper bound for normalised parameter
        
    Returns
    -------
    pM : New set of trial points
    
    """
    # perturb SOME coordinates of best population members
    nei,dim = np.shape(newP)
    M = 0
    while np.sum(M)==0:
        M     = 1*(np.random.rand(nei,dim)<=Pselect)
    pM    = newP + M*np.random.normal(0,sig,nNei)
    pM    = np.clip(pM,bL,bU)
    return pM


def GenerateNeighbours(xB,nNei,Pselect,sig,bL,bU):
    """
    Generate new set of trial parameters
    
    Parameters
    ----------
    xB : current best point
    nNei : number of neighbouring points to generate
    Pselect : probability of mutating a variable
    sig : step size for mutation
    bL : lower bound for normalised parameter
    bU : upper bound for normalised parameter
        
    Returns
    -------
    newP : New set of trial points
    
    """
    newP  = np.outer(xB,nNei*[1])
    newP  = Mutate(newP,Pselect,sig,nNei,bL,bU)
    
    return newP

def GenerateSurrogateRBF(x,f):
    """
    From current observed data and results, build a surrogate surface to predict
    returns based on parameters. Use cubic radial basis functions and polynomials
    to achieve this
    
    Parameters
    ----------
    x : list of previously evaluated parameter inputs normalised between 0 and 1
    f : list correpsonding strategy returns using parameters x
        
    Returns
    -------
    s : surrogate surface
    
    """
    n,d = x.shape
    # RBF-matrix
    R   = -2*np.dot(x,x.T) + np.sum(x**2,axis=1) + np.sum((x.T)**2,axis=0)[:,np.newaxis]
    Phi = np.sqrt(abs(R))**3             # RBF-part
    P   = np.hstack((np.ones((n,1)),x))  # polynomial part
    Z   = np.zeros((d+1,d+1))            # zero matrix
    A   = np.block([[Phi,P],[P.T,Z]])    # patched together
    # right-hand side
    F     = np.zeros(n+d+1)
    F[:n] = f                            # rhs-vector
    s = np.linalg.solve(A,F)
    return s

def EvaluateRBF(x,s,y):
    """
    Evaluate surrogate surface at desired trial point
    
    Parameters
    ----------
    x : RBF input points
    s : coefficient vector for surrogate surface
    y : trial points
        
    Returns
    -------
    ans : surrogate value at desired point
    
    """
    m,d = y.shape
    # RBF-matrix (evaluated at {y})
    R   = -2*np.dot(x,y.T) + np.sum(y**2,axis=1) + np.sum(x**2,axis=1)[:,np.newaxis]
    Phi = np.sqrt(abs(R.T))**3           # RBF-part
    P   = np.hstack((np.ones((m,1)),y))  # polynomial part
    A   = np.block([Phi,P])              # patched together
    ans = np.dot(A,s)                    # evaluation
    return ans                  

def SelectNextEvalPoint(xP,y,fy,nNei,n,n0,weights):
    """
    Select next point to perform backtest on, this is decided by a weighted average
    of the 'RBF score' (how well the surrogate predicts the point will perform)
    and the 'distance score' (how far away the point is from all previously evaluate 
    points)
    
    Parameters
    ----------
    xP : list previously evaluated points
    y : list of trial points
    fy : surrogate value at trial points
    nNei : number of trial points
    n : current number of backtests performed
    n0 : initial number of back test performed
    weights : RBF v distance weightings
        
    Returns
    -------
    x_new : Best eval point
    
    """
    V = np.zeros([nNei,1])              # Initialise total score
    deltas = np.zeros([n,nNei])         # Initialise distances matrix
    
    # Determine weightings, weights RBF score more as n increases
    w_rbf = weights[((n-n0+1)%4)-1]
    w_d = 1 - w_rbf
    
    # Calculate distance between each trial point and every xP
    for i in range(nNei):
        for j in range(n):
            deltas[j,i] = np.linalg.norm(y[i,:]-xP[j,:])
    norms = np.min(deltas,0)            # find the min distance
    # Calculate score for each trial point
    for i in range(len(fy)):
        #RBF Score
        V_rbf = (fy[i]-min(fy)) / (max(fy)-min(fy))
        # distance score
        V_d =   (max(norms)-norms[i]) / (max(norms)-min(norms))
        V[i] = w_rbf*V_rbf + w_d*V_d
    
    x_new = y[np.argmin(V)]
    
    return x_new

def UpdateCountersAndStep(f_new,fB,C_f,C_s,sig,sigmin,nV):
    """
    Update counters and step size based on whether returns from current point are
    better than previously evaluated best
    
    Parameters
    ----------
    f_new : current return value
    fB : best return value
    C_f : failure counter
    C_s : success counter
    sig : mutation step size
    sigmin : minimum step size
    nV : number of parameters
        
    Returns
    -------
    C_f : failure counter
    C_s : success counter
    sig : mutation step size
    
    """
    if f_new < fB:
        C_s = C_s + 1
        C_f = 0
    else:
        C_f = C_f + 1
        C_s = 0
    
    if C_s >= 3:
        sig *= 2
        C_s = 0
    elif C_f >= max(nV,5):
        sig *= 0.5
        C_f = 0
        if sig < sigmin:
            sig = sigmin
    
    return C_f,C_s,sig
        
def LatinHyperCube(d,m):
    """
    Generate m d-dimensional samples, distributed over a unit hypercube
    
    Parameters
    ----------
    d : number of parameters
    m : number of sample points
        
    Returns
    -------
    IPts : sample points
    
    """
    #Set up grid on hypercube in which to populate with points
    delta = np.ones(d)/m
    X     = np.zeros((m,d))
    for j in range(d):
        for i in range(m):
            X[i,j] = (2*i+1)/2*delta[j]
    P = np.zeros((m,d),dtype=int)
    
    #Generate sample points according to latin hypercube strategy
    P[:,0] = np.arange(m)
    if m%2 == 0:
        k = m//2
    else:
        k = (m-1)//2
        P[k,:] = (k+1)*np.ones((1,d))
    for j in range(1,d):
        P[0:k,j] = np.random.permutation(np.arange(k))
        for i in range(k):
            if np.random.random() < 0.5:
                P[m-1-i,j] = m-1-P[i,j]
            else:
                P[m-1-i,j] = P[i,j]
                P[i,j]     = m-1-P[i,j]
    
    IPts = np.zeros((m,d))
    for j in range(d):
        for i in range(m):
            IPts[i,j] = X[P[i,j],j]
            
    return IPts
