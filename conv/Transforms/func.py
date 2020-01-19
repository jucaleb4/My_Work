import numpy as np
import numpy.linalg as la
import scipy.linalg as scla

from math import pi,log

def expansionFMat(n,a=2,b=1):
    assert(n > 0)
    F = np.zeros((n,n))
    F[0,0] = 1
    if(n == 1):
        return F
    
    F[1,1] = 1
    prev = np.asarray([0,1])
    curr = np.asarray([-1,0,2])
    nxt = np.zeros(3)
    for j in range(2,n):
        F[:j+1,j] = curr
        
        nxt = np.zeros(j+2)
        nxt[-len(curr):] = a*curr
        nxt[:len(prev)] -= b*prev
        prev = curr
        curr = nxt
        
    return F

def evaluateBasis(B,t,i):
    """
    @B: matrix basis
    @t: time to evaluate at
    @i: the i-th polynomial
    """
    total = 0
    for j in range(i):
        total += B[j,i] * t**j
    return total

def chebyNodes(n):
    j = np.arange(n, dtype=np.float64)
    return np.cos((2*j+1)/(2*n)*np.pi)

def chebyMatrix(n):
    nodes = chebyNodes(2*n-1)

    # dim: (nodes, i)
    i = np.arange(n, dtype=np.float64)
    return np.cos(i * np.arccos(nodes.reshape(-1, 1)))

"""
Toom-Cook
"""
def rs(k):
    # produces smallest magn. ints: [1,-1,2,-2,3,...]
    return np.asarray([
        (-1)**i * (i//2 + 1) for i in range(k)
    ])

def QMat(s,n):
    assert(s >= n and s%n==0)
    z = s//n # smaller size
    w = 2*z-1 # smaller conv size
    Q = np.zeros((2*s-1, (2*n-1)*w))

    for i in range(2*n-1):
        row = i*z
        col = i*w
        Q[row:row+w, col:col+w] += np.eye(w)
        
    return Q

# creates Monomial Basis Matrix of degree n
def monoEval(n,shifted=False,cheby=False):
    z = 2*n-1
    pts = rs(z)
    if shifted:
        mdpt = (z-1)/2
        pts = (np.arange(z) - mdpt)/mdpt
    if cheby:
        i = np.arange(z, dtype=np.float64)
        pts = np.cos((2*i+1)/(2*z)*np.pi)
        
    return np.vander(pts,N=n,increasing=True)

# creates Monomial Basis Matrix of degree 2n-1
def monoInterp(n,shifted=False,cheby=False):
    z = 2*n-1
    pts = rs(z)
    if shifted:
        mdpt = (z-1)/2
        pts = (np.arange(z) - mdpt)/mdpt
    if cheby:
        i = np.arange(z, dtype=np.float64)
        pts = np.cos((2*i+1)/(2*z)*np.pi)
            
    return np.vander(pts,increasing=True)

def monomialTC(f, g, shift=False,cheby=False):
    # Toom-n => split into individual components
    n = len(f)
    
    # eval
    V = monoEval(n,shift,cheby)
    
    # multiplication
    mult = np.dot(V,f) * np.dot(V,g)

    # interpolation
    V = monoInterp(n,shift,cheby)
    coeff = la.solve(V,mult)

    # no recomposition needed
    return [V,coeff]

def TCconv2(f,g,k,shift=True,cheby=False):
    n = len(f)
    assert(n == len(g))
    r = round(log(n,k))
    assert(k**r == n)
    z = 2*k-1
        
    V = monoEval(k,shift,cheby)  
    Vs = np.eye(1)
    for i in range(r):
        Vs = np.kron(Vs,V)

    y = np.dot(Vs,f) * np.dot(Vs,g)
    
    Vinv = la.inv(monoInterp(k,shift,cheby))  
    for i in range(r-1,-1,-1):
        y = np.dot(np.kron( np.eye(z**i) , 
                            np.dot(QMat(n//k**i, k), 
                                np.kron(Vinv, np.eye(2*n//k**(i+1)-1) ) 
                                ) 
                            ),y
                        )

    return y

def chebyInterpolate(f,g):
    n = len(f)
    assert(n == len(g))
    
    z = 2*n-1
    i = np.arange(n, dtype=np.float64)
    j = np.arange(z, dtype=np.float64)

    # Chebyshev nodes:
    t = nodes = np.cos((2*j+1)/(2*z)*np.pi)

    # dim: (nodes, i)
    V = np.cos(i * np.arccos(t.reshape(-1, 1)))

    P = np.dot(V,f)
    Q = np.dot(V,g)

    # multiply
    R = P*Q

    # interpolate
    Veval = np.cos(j * np.arccos(nodes.reshape(-1, 1)))
    return la.solve(Veval, R)

def chebyTC2(f,g):
    k = len(f)
    assert(k == len(g))
    
    if(k==1): return f*g
    if(k<1): return 0

    convBack = chebyInterpolate(f,g)
    convFront = chebyInterpolate(f[::-1],g[::-1])
    convMid = np.zeros(1)
    for i in range(k):
        # only works for atomic units
        convMid += chebyTC2(f[i:i+1],g[k-i-1:k-i])
        
    return np.append(convFront[-k+1:][::-1]*2, np.append(convMid, convBack[-k+1:]*2))