#####################################################################
# Libraries
#####################################################################
import numpy as np
import pandas as pd
import scipy.interpolate as si
from tqdm import tqdm

#####################################################################
# Stochastic diff equation simulation - Milstein Method
#####################################################################
def Milstein_method( t_init: float , t_end: float , num_samples: int , init_cond: float , drift: float , diffusion: float ):

    #---------------------------------------------------#
    # (Infinitesimal) Time increment
    #---------------------------------------------------#
    dt = round(float(t_end - t_init) / num_samples , 5)
    
    #---------------------------------------------------#
    # Initial Conditions
    #---------------------------------------------------#
    X0 = init_cond

    #---------------------------------------------------#
    # dw Random process (Brownian motion)
    #---------------------------------------------------#
    def dW(Δt):
        """Random sample normal distribution"""
        return np.random.normal(loc=0.0, scale=np.sqrt(Δt))

    #---------------------------------------------------#
    # Vectors to fill
    #---------------------------------------------------#
    ts = np.arange(t_init, t_end + dt, dt)
    # ts,dt = np.linspace( t_init , t_end , num = num_samples , retstep = True )
    Xt = np.zeros( num_samples+1 )
    Xt[0] = X0

    #---------------------------------------------------#
    # Loop
    #---------------------------------------------------#
    for i in range(1, len(ts)):
        t = (i - 1) * dt
        x = Xt[i - 1]
        # Milstein method
        dw = dW(dt)
        # Xt[i] = x + drift * dt * x + diffusion * x * dw + 0.5 * diffusion**2 * x * (dw**2 - dt)
        Xt[i] = x + drift(x) * dt * x + diffusion(x) * x * dw + 0.5 * diffusion(x)**2 * x * (dw**2 - dt)

    return ts , Xt
    

#####################################################################
# Build the observations vectors with lower sample rate
#####################################################################
def Observation_sampler( t_init: float , t_end: float , step: float , ts: np.ndarray , Xt: np.ndarray ):

    x_query = np.arange( t_init , t_end , step )
    f_query = si.interpolate.interp1d( ts , Xt )   
    obs     = np.round( f_query(x_query) , 4 )

    return x_query , obs