#####################################################################
# Libraries
#####################################################################
import numpy as np
import pandas as pd
import scipy.interpolate as si
from bspline_library import *
from tqdm import tqdm

#------------------------------------------------------------------#
# Compute the estimated transition operator matrix P
#------------------------------------------------------------------#
def Probability_TransitionOperator( Xt: np.ndarray , order: int , x_init: float , x_end: float , 
                                   knots_nr: int , Nk: pd.DataFrame ):

    k_final = order
    L       = knots_nr
    p       = L+k_final-1

    knots   = np.linspace( x_init , x_end , num = L+1, endpoint=True )    # Internal knots

    P       = ss.csr_matrix( (p,p) , dtype=float )

    bspl_Xt = np.zeros( (p,len(Xt)) )
    for n in tqdm( range(len(Xt)) ):
        tmp = BSpline_querypoint( k_final , knots , Xt[n] , Nk )
        for idxTmp in range(len(tmp)):
            bspl_Xt[idxTmp,n] = np.round( tmp[idxTmp] , 8 )

    for P_row in tqdm( range(p) ):
        for P_col in range(p):
            for n in range(1,len(Xt)):
                s = bspl_Xt[P_row,n-1]*bspl_Xt[P_col,n]+bspl_Xt[P_col,n-1]*bspl_Xt[P_row,n]
                P[P_row,P_col] = P[P_row,P_col] + s
            
            P[P_row,P_col] = P[P_row,P_col]/(2*len(Xt))
    
    return P
