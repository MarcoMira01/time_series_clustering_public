#------------------------------------------------------------------#
# Libraries
#------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.sparse as ss
import scipy.linalg as sl
from tqdm import tqdm

#------------------------------------------------------------------#
# Compute B-spline basis functions with scipy.interpolate functions
#------------------------------------------------------------------#
def BSpline_basis( degree: int , support_init: float , support_end: float , num_of_basis: int ):
    #------------------------------------------------------------------#
    # Compute the B-spline basis functions for the support [x_init,x_end]
    #------------------------------------------------------------------#
    m = degree+1     # Order of the B-spline (number of constants in the polynomial)

    L = num_of_basis-m+1                                                     # Number of internal knots
    x = np.linspace( support_init , support_end , num = L+1, endpoint=True ) # Internal knots
    x = np.r_[[0]*degree, x, [x[-1]]*degree]                                 # Add boundary knots

    bspl = []
    for j in range(len(x)-m):
        # B-splibe basis computation
        bspl.append(si.BSpline.basis_element(x[j:j+m+1]))

    return bspl

#------------------------------------------------------------------#
# Basis matrix representation (Qin's matrix)
#------------------------------------------------------------------#
def BSpline_BasisMatrix_M( order: int , x_init: float , x_end: float , knots_nr: int ):
    k_final = order      # order of the polynomials
    L       = knots_nr   # Number of internal knots

    knots     = np.linspace( x_init , x_end , num = L+1, endpoint=True )    # Internal knots
    knots_ext = np.r_[[x_init]*(k_final-1) , knots , [x_end]*(k_final-1)]   # Add boundary knots

    Mk_prev = [None] * L            # Initialize Mk matrices as empty lists of length L
    Mk      = [None] * L

    # Special case k = 0
    for i in range(L):
        Mk_prev[i] = ss.csr_matrix( (1,1) , dtype=float )
        Mk_prev[i][0,0] = 1

    # k > 0
    for k in tqdm( range( 2 , k_final+1 ) ):
        for i in range(L):
            pre_Mk  = ss.csr_matrix( (k,k-1) , dtype=float )
            post_Mk = ss.csr_matrix( (k,k-1) , dtype=float )
            D0      = ss.csr_matrix( (k-1,k) , dtype=float )
            D1      = ss.csr_matrix( (k-1,k) , dtype=float )

            row = 0
            while (row < k-1):
                col = 0
                while (col < k-1):
                    pre_Mk[row,col]    = Mk_prev[i][row,col]
                    post_Mk[row+1,col] = Mk_prev[i][row,col]
                    col = col+1
                row = row+1
            
            row = 0
            i_tilde = i+k_final-1
            j = i_tilde-k+2
            while (row < k-1):
                # while (j <= i_tilde):
                if ( knots_ext[j+k-1] - knots_ext[j] != 0 ):
                    D0[row,row]   = 1-(knots_ext[i_tilde]-knots_ext[j])/(knots_ext[j+k-1]-knots_ext[j])
                    D0[row,row+1] = (knots_ext[i_tilde]-knots_ext[j])/(knots_ext[j+k-1]-knots_ext[j])
                    D1[row,row]   = -(knots_ext[i_tilde+1]-knots_ext[i_tilde])/(knots_ext[j+k-1]-knots_ext[j])
                    D1[row,row+1] = (knots_ext[i_tilde+1]-knots_ext[i_tilde])/(knots_ext[j+k-1]-knots_ext[j])
                else:
                    D0[row,row]   = 1
                    D0[row,row+1] = 0
                    D1[row,row]   = 0
                    D1[row,row+1] = 0

                j   = j+1
                row = row+1
            
            Mk[i] = (pre_Mk.dot(D0)) + (post_Mk.dot(D1))
        Mk_prev = Mk.copy()

    return Mk


#------------------------------------------------------------------#
# Selection matrix and basis matrix
#------------------------------------------------------------------#
def BSpline_BasisMatrix_N( order: int , knots_nr: int , Mk: list ):
    k_final = order
    L       = knots_nr
    p       = L+k_final-1

    Jk = [None] * L            # Initialize selection matrices Jk as empty lists of length L
    Nk = [None] * L            # Initialize basis matrices Nk as empty lists of length L

    for i in range( L ):
        Jk[i] = ss.csr_matrix( (k_final,p) , dtype=float )
        for rows in range( k_final ):
            for cols in range( i , i+k_final ):
                if ( rows+i == cols ):
                    Jk[i][rows,cols] = 1

    for i in tqdm( range(L) ):
        Nk[i] = Mk[i].dot(Jk[i])
    
    return Jk , Nk


#####################################################################
# BSpline Gram matrix
#####################################################################
def BSpline_GramMatrix( order: int , knots: np.ndarray , Nk: list ):
    k_final = order
    L       = len(knots)-1
    p       = L+k_final-1

    Delta_mat = ss.csr_matrix( (k_final,k_final) , dtype=float )
    for row in range(k_final):
        for col in range(k_final):
            Delta_mat[row,col] = 1/(col+row+1)

    Gram_mat = ss.csr_matrix( (p,p) , dtype=float )
    for i in tqdm( range(L) ):
        Delta_mat_i  = (knots[i+1]-knots[i])*Delta_mat
        # Delta_mat_i  = Delta_mat
        Nk_tmp       = Nk[i]
        Nk_transpose = Nk[i].transpose()
        tmp          = Nk_transpose*Delta_mat_i*Nk_tmp
        Gram_mat = Gram_mat + tmp

    return Gram_mat

#####################################################################
# BSpline Cholesky orthogonalization
#####################################################################
def BSpline_CholOrthogon( Gram_mat: ss.csr_matrix , Nk_tilde: list , int_knots_nr: int ):
    L = int_knots_nr

    lower = np.round( sl.cholesky(np.round( Gram_mat.toarray() , 7 ), lower=True) , 6 )

    LambdaT  = ss.csr_matrix( np.round( np.transpose( np.linalg.inv(lower) ) , 6 ) )

    Nk = [None] * L
    for i in range(L):
        Nk[i] = ( Nk_tilde[i].dot( LambdaT ) )

    return Nk

#####################################################################
# Compute BSpline set in a query point
#####################################################################
def BSpline_querypoint( order: int , knots: np.ndarray , query_point: float , Nk: list ):
    k = order
    L = len(knots)-1
    p = L+k-1

    x_q = query_point   # query point

    # Interval identification
    i = 1
    flag = False
    while ( i <= L ) and ( flag == False ):
        if ( x_q <= knots[i] ):
            flag = True
            idxItv = i-1
        i = i+1
    
    # Compute the basis in x_q
    U = ss.csr_array( (k,1) , dtype=float )
    u = np.round( (x_q-knots[idxItv])/(knots[idxItv+1]-knots[idxItv]) , 8 )
    for j in range( k ):
        U[j,0] = np.power( u , j )

    Nk_T = Nk[idxItv].transpose( )
    tmp  = Nk_T.dot(U)
    bspl = tmp.toarray( )

    return bspl

#####################################################################
# BSpline curves
#####################################################################
def BSpline_curves( order: int , knots: np.ndarray , Nk: list ):
    k = order
    L = len(knots)-1
    p = L+k-1

    t_ivl = np.linspace(knots[0], knots[-1], num = (100*L)+1 , endpoint=True)
    bspl  = np.zeros( [p , len(t_ivl)] , dtype=float )

    # B-spline basis computation
    for col in tqdm( range( len(t_ivl) ) ):
        bspl_tmp = BSpline_querypoint( k , knots , t_ivl[col] , Nk )

        for row in range( p ):
            bspl[row,col] = bspl_tmp[row]

    return t_ivl , bspl

#####################################################################
# BSpline plot
#####################################################################
def BSpline_plot( t_ivl: np.ndarray , bspl: np.ndarray ):
    
    plt.figure(figsize=(5, 5))
    
    dim = bspl.shape

    for row in range( dim[0] ):
        toplot = bspl[row,0:]
        plt.plot( t_ivl , toplot )

    plt.show()