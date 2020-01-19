import numpy as np
import numpy.linalg as la
from math import log,ceil,floor
from scipy import linalg as scila
from sympy import Poly
from sympy.polys import ring, QQ
RR, x = ring("x", QQ)

# setup code
def randomvec(n, seedNumber = 0):
    np.random.seed(seedNumber)
    return np.random.rand(n)

def vectorToToeplitz(v,N=-1):
    band_width = len(v)
    if(N <= 0):
        N = band_width

    assert(band_width <= N)

    H = np.zeros((N * 2 - 1, N))
    for col in range(N):
        H[col:col+band_width , col] = v
    return H

def toeplitzToHankle(M):
    return M[:,::-1]

def createHankelMatrix(width, band, tri=True, square=True):
    band_width = len(band)
    assert(band_width >= width)

    t = vectorToToeplitz(band, width)
    if(not tri):
        y = randomvec(band_width, seedNumber=4)
        t2 = vectorToToeplitz(y, width)
        t[:width] += t2[:width].T
    h = toeplitzToHankle(t)
    return h[:width] if square else h

def randomHankel(width, band_width=-1, tri=True, square=True):
    if(band_width == -1): band_width = width
    return createHankelmatrix(width, randomvec(band_width), tri, square)

## For FFT
def omega(n):
    return np.exp(-2*np.pi*1j/n)

def Fmat(n, N=-1, offset=0):
    w = omega(n if N==-1 else N)
    F = np.zeros((n,n), dtype=complex)
    for i in range(n):
        for j in range(n):
            F[i,j] = w**(i*(j+offset))
    return F

def Fmatinv(n):
    w = omega(n)
    Finv = np.zeros((n,n), dtype=complex)
    for i in range(n):
        for j in range(i,n):
            Finv[i,j] = Finv[j,i] = w**(-1*i*j)/n
    return Finv

def fft(v, twiddle=-1):
    n = v.size

    if n is 1:
        return v

    u = fft(v[::2], twiddle) # compute FFT of [v_0,v_2,...v_{n-1}]
    w = fft(v[1::2], twiddle) # compute FFT of [v_1,v_3,...v_n]

    # scale w by twiddle factors [omega_(n)^0,...omega_(n)^(n/2-1)]
    twiddle = n if twiddle == -1 else twiddle
    t = np.asarray([omega(twiddle)**i for i in range(n//2)]) 
    z = w*t

    return np.concatenate([u+z,u-z])

def invfft(v):
    return invfftHelper(v)/v.size

def invfftHelper(v):
    n = v.size
    if n is 1:
        return v
    u = invfftHelper(v[::2]) # compute FFT of [v_0,v_2,...v_{n-1}]
    w = invfftHelper(v[1::2]) # compute FFT of [v_1,v_3,...v_n]

    # scale w by twiddle factors [omega_(n)^0,...omega_(n)^(n/2-1)]
    t = np.asarray([omega(n)**(-i) for i in range(n//2)])
    z = w*t

    return np.concatenate([u+z,u-z])

def createCirculantMatrix(T):
    n = T.shape[0]

    C = np.zeros((2*n,2*n))
    C[:n,:n] = T.copy()
    C[n:2*n,n:2*n] = T.copy()

    T2 = np.zeros((n,n))
    lower = T[0,1:].copy()[::-1]
    upper = T[1:,0].copy()[::-1]
    for i in range(1,n):
        T2[i-1,i:n] = upper[:n-i]
        T2[i:n,i-1] = lower[:n-i]
    C[:n,n:2*n] = T2.copy()
    C[n:2*n,:n] = T2.copy()

    return C

def circulantColumn(col,row):
    """
    total will be 2n-1 DOF
    """
    n = len(row)
    assert(n == len(col))
    
    return np.append(np.append(col, 0), row[1:][::-1])

## FRHA

# https://www.geeksforgeeks.org/rotate-bits-of-an-integer/
# Python3 code to 
# rotate bits of number 

# Function to left rotate n by d bits 
def leftRotate(val, n, b): 
    assert(n >= b >= 0)
    assert(n > val)
    bits = int(log(n)/log(2))
    shift = int(log(b)/log(2))
    return ((1 << bits) - 1) & (val << shift) | (val >> (bits - shift)) 

# Function to right rotate n by d bits 
def rightRotate(val, n, b): 
    assert(n >= b >= 0)
    assert(n > val)
    bits = int(log(n)/log(2))
    shift = int(log(b)/log(2))
    return (val >> shift) | (val << (bits - shift)) & ((1 << bits) - 1)

def bitRotateVec(v,k,left=True):
    n = len(v)
    vRotate = np.zeros(n)
    for i in range(n):
        if left:
            vRotate[i] = v[leftRotate(i,n,k)]
        else:
            vRotate[i] = v[rightRotate(i,n,k)]
    return vRotate

def hankelImplicitForm(H):
    return np.append(H[0,:-1],H[:,-1])

###########################
#### FRHA with borders ####
###########################
def reorgHankelvecWithBounds(band,hlen,l,r):
    """
    @band - band elements of b size
    @hlen - original hankel size
    @l    - left boundary (inclusive)
    @r    - right boundary (inclusive)
    """
    
    # fix edge case
    origin = l % 2
    reorderedBlocks = [
            band[0 + origin::2],
            band[1 - origin::2],
            band[2 - origin::2]
        ]
    
    sizeLimit = 2*hlen-1
    newBoundaries = [
        [ ceil(l/2), min(floor(r/2), sizeLimit-1) ],
        [ floor(l/2), ceil(r/2)-1 ],
        [ max(ceil(l/2)-1,0), floor(r/2)-1 ]
    ]
    
    return [reorderedBlocks, newBoundaries]

def addHankelvec(band1,bounds1,band2,bounds2):
    assert(len(bounds1) == 2 and len(bounds2) == 2)
    
    leftBound  = min(bounds1[0],bounds2[0])
    rightBound = max(bounds1[1],bounds2[1])
    sumBand = np.zeros(rightBound-leftBound+1)
    
    band1Start = bounds1[0] - leftBound
    sumBand[band1Start:band1Start+len(band1)] = band1
    band2Start = bounds2[0] - leftBound
    sumBand[band2Start:band2Start+len(band2)] += band2
    
    return [sumBand,leftBound,rightBound] 

# Implicit Tridiagonal Hankel MUltipliction and Boundaries
def ITHUMB(band,x,impLen,l,r):
    bandLen = r-l+1
    assert(bandLen <= 3 and bandLen == len(band))
    
    n = len(x)
    
    ans = np.zeros(n)
    x = np.append(np.append(np.zeros(2),x), np.zeros(2))
    
    offset = 0
    # lower triangle
    if(l == impLen//2):
        offset = 1
    # upper triangle
    elif(r == impLen//2):
        offset = -1
    
    for i in range(n):
        top = n+offset-i
        bot = n+offset+bandLen-i
        ans[i] = np.inner(band, x[top:bot])
        
    return ans

def _FIREHAB(band,x,l,r):
    """
    @band   - band elements of b size
    @x      - vector to convolve with
    @l      - left boundary (inclusive)
    @r      - right boundary (inclusive)
    """
    assert(l <= r)
    n = len(x) # also the size of the hankel matrix
    
    if(r-l+1 <= 3):
        return ITHUMB(band, x, 2*n-1, l, r)
    
    [subBlks,bounds] = reorgHankelvecWithBounds(band,n,l,r)
    x = bitRotateVec(x, k=2**1, left=True)
    
    x1 = x[:n//2]; x2 = x[n//2:]
    a,b,c = subBlks

    h1,l1,r1 = addHankelvec(a,bounds[0],b,bounds[1])
    h2 = b; l2 = bounds[1][0]; r2 = bounds[1][1]
    h3,l3,r3 = addHankelvec(b,bounds[1],c,bounds[2]) 
    
    y1 = _FIREHAB(h1, x1   , l1, r1)
    y2 = _FIREHAB(h2, x1-x2, l2, r2)
    y3 = _FIREHAB(h3, x2   , l3, r3)
    
    return bitRotateVec(np.append(y1-y2, y2+y3), k=2**1, left=False)

# Fast Implicit REorganized Hankel Algorithm and Boundaries
def FIREHAB(band,x):
    """
    @band: band elements, must be from lower triangular matrix
    @x:    vector to be convolved
    """
    n = len(x)
    b = len(band)
    return _FIREHAB(band,x,n-1,n+b-2)

"""
Toom-Cook
"""
def rs(k):
    # produces smallest magn. ints: [1,-1,2,-2,3,...]
    return np.asarray([
        (-1)**(i) * ceil(i/2) for i in range(k)
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

def chebyInterpolate2(f,g):
    n = len(f)
    assert(n == len(g))
    
    z = 2*n-1
    N = 2*n
    Z = 2*N-1
    i = np.arange(N, dtype=np.float64)
    j = np.arange(Z, dtype=np.float64)

    # Chebyshev nodes:
    t = nodes = np.cos((2*j+1)/(2*Z)*np.pi)

    # dim: (nodes, i) (4n-1 x n)
    V = np.cos(i * np.arccos(t.reshape(-1, 1)))
    V = V[:,n:]

    p = np.dot(V,f)
    q = np.dot(V,g)

    # multiply
    r = p*q

    # interpolate (2n-1 x 4n-1)
    Veval = np.cos(j * np.arccos(nodes.reshape(-1, 1)))
    Vinv = la.inv(Veval)
    # print("Norm:",la.norm(Vinv @ Veval))
    Vinv = Vinv[N:,:]
    return np.dot(Vinv, r)*2

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

def Toom_k(f,g,k,E,I):
    n = len(f)
    assert(n == len(g))
    p = round(log(n,k))
    assert(k**p == n)
    
    if n == 1: return f*g
    
    # fold into a quasi-tensor
    F = np.reshape(f, newshape=(k,-1))
    G = np.reshape(g, newshape=(k,-1))
    
    # evaluate 
    P = np.dot(E,F)
    Q = np.dot(E,G)
    x,y = P.shape
    
    # multiplication (recursion)
    R = np.zeros((x,2*y-1))
    for i in range(x):
        R[i] = Toom_k(P[i],Q[i],k,E,I)
        
    # interpolate
    T = np.dot(I,R)

    # recompose
    t = np.reshape(T, newshape=(-1,)) 
    recomp = QMat(n,k)
    return np.dot(recomp, t)

def chebyTCMats(k,low=False):
    if not low: k *= 2
    i = np.arange(k, dtype=np.float64)
    j = np.arange(2*k-1, dtype=np.float64)
    nodes = np.cos((2*j+1)/(2*(2*k-1))*np.pi)
    if not low: k = k//2

    E = np.cos(i * np.arccos(nodes.reshape(-1, 1)))
    if not low: E = E[:,k:]

    # interpolate
    I = np.cos(j * np.arccos(nodes.reshape(-1, 1)))
    I = la.inv(I)
    if not low: I = I[2 * k:,:]
    
    return [E,I]


"""
Winograd
"""
# assume N=1 image, K=1 filter, C=1 channel
def simpleWinogradAlg(g,d,m,B,G,A):
    N = K = C = 1
    """
    @g: 2d.numpy_array as square filter
    @d: 2d.numpy_array as data
    @m: int as output of FIR filter F(m,r)
    """
    h,w = d.shape
    r = g.shape[0]
    
    assert(g.shape[0] == g.shape[1])
    assert(h%m == 0 and w%m == 0)
    
    h-=m; w-=m
    
    P = (h//m)*(w//m) # num of tiles
    a = m+r-1 # input tile size
    
    dChunks = np.zeros((C,P,a,a))
    for c in range(C):
        for y in range(h//m):
            for x in range(w//m):
                b = y*(w//m) + x
                dChunks[c,b] = d[(y*m):(y*m)+a, (x*m):(x*m)+a]
    
    U = np.zeros((a,a,K,C))
    for k in range(K):
        for c in range(C):
            uInterm = np.dot(G, np.dot(g, G.T))
            for e in range(a):
                for v in range(a):
                    U[e,v,k,c] = uInterm[e,v]
            
    V = np.zeros((a,a,C,P))
    for b in range(P):
        for c in range(C):
            vInterm = np.dot(B.T, np.dot(dChunks[c,b], B))
            for e in range(a):
                for v in range(a):
                    V[e,v,c,b] = vInterm[e,v]
            
    M = np.zeros((a,a,K,P))
    for e in range(a):
        for v in range(a):
            M[e,v] = np.dot(U[e,v], V[e,v])
            
    Y = np.zeros((K,P,m,m))
    for k in range(K):
        for b in range(P):
            mInterm = np.zeros((a,a))
            for e in range(a):
                for v in range(a):
                    mInterm[e,v] = M[e,v,k,b]         
            Y[k,b] = np.dot(A.T, np.dot(mInterm, A))
        
    Ynew = np.zeros((K,h,w))
    for k in range(K):
        for y in range(h//m):
            for x in range(w//m):
                b = y*(w//m) + x
                Ynew[k,y*m:(y+1)*m, x*m:(x+1)*m] = Y[k,b]
    return Ynew

def padImage(g,r):
    h,w = g.shape
    g2 = np.zeros((2*r-2 + h,2*r-2 + w))
    g2[r-1:r-1+h,r-1:r-1+w] = g
    return g2

def revMatrix(M):
    n1,n2 = M.shape
    return np.eye(n1)[::-1] @ M @ np.eye(n2)[::-1]


"""
Doubly Toeplitz Matrix
"""
def quasiVec(V, blkSize):
    m,n = V.shape
    return np.reshape(V[::-1], newshape=(m*n,))

def vecToConvMatrix(v, shape):
    assert(len(shape) == 2)
    return np.reshape(v, newshape = shape)[::-1]

def createDoublyToeplitz(F,fm,fn,gm,gn):
    m = fm+gm-1; n = fn+gn-1
    
    F2 = np.zeros((m,n)).copy()
    F2[m-fm:,:fn] = F.copy()
    F = F2
    
    Fs = np.zeros((m,n,gn))
    diffZero = n-fn
    
    Ts = np.zeros((m,n,gn))
    for y in range(m):
        smallT = vectorToToeplitz(F[y])[:n,:gn]
        Ts[m-y-1] = smallT.copy()
    
    Tfull = np.zeros((m*n,gm*gn))
    for i in range(m):
        sel = i
        for j in range(gm):
            Tfull[i*n:(i+1)*n, j*gn:(j+1)*gn] = Ts[sel]
            sel -= 1
            if(sel < 0):
                sel = m-1
                
    return Tfull
            
def convolve2DToeplitz(F,G):
    fm,fn = F.shape; gm,gn = G.shape
    Tfull = createDoublyToeplitz(F,fm,fn,gm,gn)
    g = quasiVec(G,gm)
    vecTConv = np.dot(Tfull, g)
    m = fm+gm-1; n = fn+gn-1
    return vecToConvMatrix(vecTConv, (m,n))


#### Methods for building generic algorithms ####

def toomCookMats(r, n, cheby=False):
    pts = rs(n+r-1)
    V = np.vander(pts,increasing=True)
    V[-1,:] = 0; V[-1,-1] = 1
    C = la.inv(V)
    
    A = V[:,:r].copy(); A[-1,-1] = 1
    B = V[:,:n].copy(); B[-1,-1] = 1
    
    return [C,A,B]

def nestedToomCook(n):
    temp = n
    primes = np.asarray([2,3,5,7])
    scheme = np.asarray([])
    while(temp > 1):
        found = False
        for p in primes:
            if(temp % p == 0):
                temp /= p
                scheme = np.append(scheme,p)
                found=True
                break
        if(not found): 
            print("invalid prime scheme")
            assert(False)
    CC = np.asarray([1])
    BB = np.asarray([1])
    AA = np.asarray([1])
    
    total_size = 1
    for k in scheme:
        k = int(k)
        total_size *= k
        [C,A,B] = toomCookMats(k,k)
        Q = QMat(total_size,k)
        CC = np.dot(Q,np.kron(CC,C))
        AA = np.kron(AA,A)
        BB = np.kron(BB,B)
        
    return [CC,AA,BB]

def fftMats(b, n, cyclic=False):
    assert(b <= n)
    z = (n if cyclic else n+b-1)
    F = Fmat(z)
    Finv = Fmatinv(z)
    
    return [F[:,:n].copy(),F[:,:b].copy(),Finv]

def sym_deg(p):
    return p.terms()[0][0][0]

def sym_allcoeffs(p):
    # extracts all coefficients form 1 -> x**p
    d = sym_deg(p)
    coeffs = np.zeros(d+1)
    terms = p.to_dict()
    for i in range(d+1):
        v = tuple([i])
        if v in terms:
            coeffs[i] = terms[v]
    return coeffs

def extEucAlg(M,m,RR,deg):
    # returns coefficients of N(x) and n(x) 
    [N,n,gcd] = RR.dup_gcdex(M, m)
    assert(len(gcd) == 1 and gcd.coeffs()[0] == 1)
    # !!! <!-- TODO --> sum of degrees not right, this is an ad-hoc fix
    if(sym_deg(M) + sym_deg(N) != deg):
        Ntemp = N
        N = n*m
        n = Ntemp*M
    return [sym_allcoeffs(N),sym_allcoeffs(n)]

def berzotMats(pTerms,numCoeffs):
    deg = len(pTerms)-1
    E = np.zeros((numCoeffs + deg, numCoeffs))
    for i in range(deg):
        for j in range(deg+1):
            E[j+i,i] = pTerms[j]
    return E

def moduloMats(divPoly,deg):
    # build dividend polynomial matrix
    divMat = np.eye(deg+1)
    
    # divisor vector
    cffs = sym_allcoeffs(divPoly)
    divDeg = len(cffs)-1

    for i in range(deg-divDeg+1):
        idx = deg-i
        row = divMat[idx,:]
        prod = np.outer(cffs,row)
        divMat[idx-divDeg:idx+1,:] -= prod
        
    return divMat[:divDeg,:]

def cyclicWinoMats(d,RR,x):
    m1 = x**(d//2) + 1
    m2 = x**(d//2) - 1
    
    d1,d2 = d//2,d//2
    
    # evaluates our input d-point, or (d-1)-degree
    # in the polynomial modulo M,m
    M1 = moduloMats(m1,d-1)
    M2 = moduloMats(m2,d-1)

    # split nesting uses Agarawal to derive larger small convs
    C,A,_ = toomCookMats(d1,d1)
    A1 = np.dot(A,M1)
    A2 = np.dot(A,M2)
    AA = np.vstack([A1,A2])

    # evaluates the result degree (2n-1)-1 in poly. modulo M,m again
    M3 = moduloMats(m1,2*d1-2)
    M4 = moduloMats(m2,2*d2-2)

    C1 = np.dot(M3,C)
    C2 = np.dot(M4,C)
    CC = scila.block_diag(C1,C2)

    # linear combinations by Berzouts identity for the CRT
    N,n = extEucAlg(m1,m2,RR,d)
    E1 = berzotMats(N,d1)
    E2 = berzotMats(n,d2)
    E = np.hstack([E1,E2])
    EC = np.dot(E,CC)
    
    # [C,A,B]
    return [EC,AA,AA]

def _buildModuloMat(modPoly,divMat):
    # divisor as vector of coefficients
    deg = len(divMat)-1
    poly = sym_allcoeffs(modPoly)
    polyDeg = len(poly)-1

    # long division
    for i in range(deg-polyDeg+1):
        idx = deg-i
        row = divMat[idx,:]
        prod = np.outer(poly,row)
        divMat[idx-polyDeg:idx+1,:] -= prod
        
    return divMat[:polyDeg,:]

# Input(polynomial modulo, 
# to-be evaluated d-degree polynomial - can be arbitrary
# if genericPoly is empty
#       [optional]generic starting polynomial)
# Output: Matrix of a polynomial of degree (d-1) in modulo space
# I.e. Evaluate (P(x) mod M) (mod m_i)
# Note: polynomial in modulo poly (d+1)-degree evals to d-degree
def buildModuloMat(modPoly,deg,genericPoly=None):
    if(genericPoly is None):
        # d-degree has d+1 coefficients
        # where ith variable goes to power x^i
        genericPoly = np.eye(deg+1)
    return _buildModuloMat(modPoly,genericPoly)

# Input(idompotent poly, to-be multiplied d-degree polynomial)
# Output: Matrix of polynomial product (no modulo)
def _buildMultMat(idePoly,deg):
    ideCoeff = sym_allcoeffs(idePoly)
    ideDeg = len(ideCoeff)-1
    
    # has same number of deg+1 variables, but power
    # will be extended to deg+ideDeg
    prodMat = np.zeros((deg+ideDeg+1,deg+1))
    
    # multiply by coefficient of x^i
    for i in range(ideDeg+1):
        prodMat[i:i+deg+1,:] += ideCoeff[i] * np.eye(deg+1)
    return prodMat

# Input(idempotent polynomial coeffs,
#        large modulo polynomial,small mod space poly)
# Output: Matrix of evaluated product in modulo space m
def buildIdeMat(idePoly,M,mi):
    deg_m = sym_deg(mi)
    deg_M = sym_deg(M)
    # recall given (d+1)-degree mod poly, evals d-degree poly
    prodMat = _buildMultMat(idePoly,deg_m-1)
    ideMat = buildModuloMat(M,None,prodMat)
    return ideMat

def getBerzetPolys(M,m,RR,deg):
    # returns coefficients of N(x) and n(x) 
    [N,n,gcd] = RR.dup_gcdex(M, m)
    assert(len(gcd) == 1 and gcd.coeffs()[0] == 1)
    return [N,n]
    # return [sym_allcoeffs(N),sym_allcoeffs(n)]

def vectorToToeplitz2(v,numCols):
    band_width = len(v)
    H = np.zeros((numCols+len(v)-1, numCols))
    for col in range(numCols):
        H[col:col+band_width , col] = v
    return H
    
def buildModuloMat2(modPoly,deg):
    poly = sym_allcoeffs(modPoly)
    polyDeg = len(poly)-1
    if(polyDeg > deg):
        I = np.zeros((polyDeg,deg+1))
        I[:deg+1,:deg+1] = np.eye(deg+1)
        return I
    T = vectorToToeplitz2(poly,deg-polyDeg+1)
    K = T[:polyDeg,:]
    L = T[polyDeg:]
    E = np.hstack([np.eye(polyDeg), -np.dot(K,la.inv(L))])
    return E

# returns N,n coefficients
def getBerzetPolys2(M,m):
    cM = sym_allcoeffs(M)
    cm = sym_allcoeffs(m)
    dM = len(cM)-1
    dm = len(cm)-1
    TM = vectorToToeplitz2(cM,dm)
    Tm = vectorToToeplitz2(cm,dM)
    T = np.hstack([TM,Tm])
    coeffs = la.solve(T,np.append(1,np.zeros(dm+dM-1)))
    return [coeffs[:dm],coeffs[dm:]]
    
def getProdPoly(polys):
    prod = 1
    for p in polys:
        prod *= p
    return prod

"""
def winoMats(polys,RR,x):
    M = getProdPoly(polys)
    deg_M = sym_deg(M)
    
    AA = None
    CC = None
    EE = None
    first = True
    
    for mi in polys:
        deg_m = sym_deg(mi)
        
        # evaluate in modulo
        L = buildModuloMat2(mi,deg_M-1)
        C,A,_ = toomCookMats(deg_m,deg_m)
        A = np.dot(A,L)
        AA = (A if first else np.vstack([AA,A]))

        # evaluates interpolation modulo
        L = buildModuloMat2(mi,2*deg_m-2)
        C = np.dot(L,C)
        CC = (C if first else scila.block_diag(CC,C))

        # recovery by the CRT
        Mi = M/mi
        cMi = sym_allcoeffs(Mi)
        N,n = getBerzetPolys2(Mi,mi)
        TcM = vectorToToeplitz2(cMi,len(cN))
        cE = np.dot(TcM,cN)
        E = vectorToToeplitz2(cE,2*deg_m+1)
        EE = (E if first else np.hstack([EE,E]))
        first = False
            
    EC = np.dot(EE,CC)
    return [EC,AA,AA]
"""

def winoMats(polys,n,r,nest=False):
    M = getProdPoly(polys)
    deg_M = sym_deg(M)
    
    AA = None
    BB = None
    CC = None
    EE = None
    first = True
    
    for mi in polys:
        deg_m = sym_deg(mi)
        
        # evaluate in modulo
        if(nest):
            C,A,B = nestedToomCook(deg_m)
        else:
            C,A,B = toomCookMats(deg_m,deg_m)
        X = buildModuloMat2(mi,r-1)
        A = np.dot(A,X)
        AA = (A if first else np.vstack([AA,A]))
        X = buildModuloMat2(mi,n-1)
        B = np.dot(B,X)
        BB = (B if first else np.vstack([BB,B]))

        # evaluates interpolation modulo, recall we will have
        # two degree deg_m-1
        X = buildModuloMat2(mi,2*deg_m-2)
        C = np.dot(X,C)
        CC = (C if first else scila.block_diag(CC,C))

        # recovery by the CRT
        Mi = M/mi
        cMi = sym_allcoeffs(Mi)
        cmi = sym_allcoeffs(mi)
        cN,_ = getBerzetPolys2(Mi,mi)
        TcM = vectorToToeplitz2(cMi,len(cN))
        cE = np.dot(TcM,cN)
        E = vectorToToeplitz2(cE,deg_m)
        
        # evaluate modulo M
        X = buildModuloMat2(M,len(E)-1)
        E = np.dot(X,E)
        
        EE = (E if first else np.hstack([EE,E]))
        first = False
            
    EC = np.dot(EE,CC)
    return [EC,AA,BB]

def recoverLinConv(polys,_y,fLast,gLast):
    M = getProdPoly(polys)
    Mcoeffs = sym_allcoeffs(M)
    y = np.zeros(len(_y) + 1)
    y[:-1] += _y
    y[-len(Mcoeffs):] += fLast * gLast * Mcoeffs
    return y

def moduloToLinear(polys,n,r,C,A,B):
    p = getProdPoly(polys)
    pc = sym_allcoeffs(p)
    C2 = np.zeros((len(pc),C.shape[1]+1))
    C2[:C.shape[0],:C.shape[1]] = C.copy()
    C2[:,C.shape[1]] = pc.copy()
    A2 = np.vstack([A,np.zeros(A.shape[1])])
    A2 = A2[:,:r]
    B2 = np.vstack([B,np.zeros(B.shape[1])])
    B2 = B2[:,:n]
    A2[-1,-1] = 1
    B2[-1,-1] = 1
    return [C2,A2,B2]

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