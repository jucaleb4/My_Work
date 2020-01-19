import csv
import numpy as np
import numpy.linalg as la
import scipy.linalg as spla
from math import log

def getPrices(ticker):
    closeIdx = 0
    firstRow = True
    firstDate = True
    prices = []
    dates = []
    
    with open('individual_stocks_5yr/' + ticker + '_data.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(spamreader):
            if(firstRow):
                for i, col in enumerate(row):
                    if(col == 'close'):
                        closeIdx = i; firstRow = False
                        continue;
            else:
                dates.append(row[0])
                prices.append(float(row[closeIdx]))
    return dates, prices

def vandermonde(n):
    return np.array([[t**i for i in range(n)] for t in range(n)])

def singlePolynomialLinSys(t,y):
    if(len(t) != 2):
        print("Error, need size 2 for time, got size {}".format(len(t)))
        assert(False)
    if(len(y) != 2):
        print("Error, need size 2 for y's, got size {}".format(len(y)))
        assert(False)
        
    A = np.zeros((2,4))
    
    for i in range(2):
        A[i,:4] = [t[i]**j for j in range(4)]
    
    return [A, np.array([y[0],y[1]])]

def singleDerivativesLinSys(t):
    A = np.zeros((2,8))

    # 1-deriv
    for i in range(1,4):
        A[0,i] = i * t**(i-1)
    # 2-driv
    A[1,2] = 2; A[1,3] = 6 * t
    # equalize
    A[1:,5:8] = -A[1:,1:4]
    
    return [A,np.zeros(2)]

def cubicSplineLinSys(v,time):
    n = len(v)
    
    N = 4*(n-1)
    A = np.zeros((N,N))
    b = np.zeros(0)
    
    # polynomials
    for i in range(n-1):
        t = time[i]; t2 = time[i+1]
        [AA, bb] = singlePolynomialLinSys([t,t2], v[i:i+2])
        A[i*2:(i+1)*2,i*4:(i+1)*4] = AA
        b = np.append(b,bb)
        
    offset = 2*(n-1)
    # derivatives
    for i in range(n-2):
        t1 = time[i+1]
        [AA, bb] = singleDerivativesLinSys(t1)
        A[i*2+offset: (i+1)*2+offset, i*4:(i+2)*4] = AA
        b = np.append(b,bb)
        
    # natural spline
    A[-2,2:4] = [2,0] # [2, 6*t_1]
    A[-1,-2:] = [2,6*time[n-1]] # [2, 6*t_{n-1}]
    b = np.append(b, np.zeros(2))
    
    return [A,b]

def cubicSplineCoeff(points, time):
    assert(len(points) == len(time))
    A,b = cubicSplineLinSys(points, time)
    [Q,R] = la.qr(A)
    return spla.solve_triangular(R,np.dot(Q.T, b))

def contCubicSplineLinSys(v,dv1,dv2,time):
    n = len(v)
    assert(n == len(time))
    N = 4*(n-1)
    A = np.zeros((N,N))
    b = np.zeros(0)
    
    # polynomials
    for i in range(n-1):
        # print(i,n)
        t = time[i]; t2 = time[i+1]
        [AA, bb] = singlePolynomialLinSys([t,t2], v[i:i+2])
        A[i*2:(i+1)*2,i*4:(i+1)*4] = AA
        b = np.append(b,bb)
        
    offset = 2*(n-1)
    # derivatives
    for i in range(n-2):
        t1 = time[i+1]
        [AA, bb] = singleDerivativesLinSys(t1)
        A[i*2+offset: (i+1)*2+offset, i*4:(i+2)*4] = AA
        b = np.append(b,bb)
        
    # continous spline
    A[-2,1:4] = [1,2*time[0],3*time[0]**2] # [1, 2*t_1, 3*t_1^2]
    A[-1,-3:] = [1,2*time[n-1],3*time[n-1]**2] # [1, 2*t_n, 3*t_n^2]
    b = np.append(b, np.array([dv1,dv2]))
    
    return [A,b]

def continousSplineCoeff(p,time,maxSize=100):
    assert(len(p) == len(time))
    n = 0
    coeff = np.zeros(0)
    maxSize = min(maxSize, len(p))
        
    while(n < len(p)):
        if(len(p) - n - maxSize <= 4):
            maxSize += len(p) - n

        maxSize = min(maxSize, len(p) - n)
            
        dv1 = (p[n+1] - p[n])/(time[n+1] - time[n])
        dv2 = (p[n+maxSize-1] - p[n+maxSize-2])/(time[n+maxSize-1] - p[n+maxSize-2])
        
        A,b = contCubicSplineLinSys(p[n:n+maxSize+1],dv1,dv2,time[n:n+maxSize+1])
        Q,R = la.qr(A)
        coeff = np.append(coeff, spla.solve_triangular(R, np.dot(Q.T, b)))
        
        n += maxSize
    
    return coeff

def evaulateCubicSpline(coeff, xCoor, evalPts, order=0):
    # ensures 4(n-1) degrees of freedom
    assert(len(coeff) == 4 * (len(xCoor) - 1))
    
    ys = np.zeros(0)
    splineIdx = 0
    
    for t in evalPts:
        while not (xCoor[splineIdx//4] <= t <= xCoor[splineIdx//4 + 1]):  
            splineIdx += 4
            if splineIdx >= len(coeff)-3:
                print("Error: Evaluation point {} outside of spline endpoint {}".\
                     format(t, xCoor[-1]))
                assert(False)  
        
        if order == 0:
            val = coeff[splineIdx] + coeff[splineIdx+1]*t \
                + coeff[splineIdx+2]*t**2 + coeff[splineIdx+3]*t**3
        elif order == 1:
            val = coeff[splineIdx+1] + 2*coeff[splineIdx+2]*t + \
                    3*coeff[splineIdx+3]*t**2
        elif order == 2:
            val = 2*coeff[splineIdx+2] + 6*coeff[splineIdx+3]*t
        else:
            print("Error: order must be between [0,2], got {}".format(order))
            assert(False)
            
        ys = np.append(ys, val)
        
    return ys

def omega(n):
    return np.exp(-2*np.pi/n)

def fourierMatrix(n,col=0,row=0):
    F = np.zeros((n-col,n-row))
    for i in range(n-col):
        for j in range(n-row):
            F[i,j] = omega(n)**(i*j)
    return F

def DFT(v):
    return fourierMatrix(len(v)) @ v

# def extractCP(t,v,dv,threshold=2.5e-1):
#     if(len(t) != len(v) or len(t) != len(dv)):
#         print("Error: dimensions t,v,dv must match. Got {}, {}, {} respectively"\
#               .format(len(t), len(v), len(dv)))
#         assert(False)
    
#     # CP -> critical point
#     tt = np.zeros(0); vv = np.zeros(0)
#     for i in range(len(t)):
#         if abs(dv[i]) < threshold:
#             tt = np.append(tt,t[i])
#             vv = np.append(vv,v[i])
            
#     return [tt,vv]

def newtons(c0,c1,c2,c3,a,b,tol,maxTries):
    assert(a < b)
    x_prev = b
    x = a
    
    for _ in range(maxTries):
        if(x < a or x > b):
            return [False,-1,1] # exitted domain
        
        elif(abs(x - x_prev) >= tol):
            x_prev = x

            df1 = c1 + 2*c2*x + 3*c3*x**2 # 1+2t+3t^2
            df2 = 2*c2 + 6*c3*x # 2+6t
            
            if df2 == 0: return [False,-1,0] # inflection pt
            x = x - df1.real/df2.real
            
        else:
            val = c0 + c1*x + c2*x**2 + c3*x**3
            return [True,x,val] # found
        
    return [False,-1,0]

def findCP(coeff, time, tol=1e-1,localMaxOnly=False, maxTries=5):
    assert(len(coeff) == 4*(len(time)-1))
    
    numPts = len(coeff)//4
    exitTimes = 0

    tt = np.zeros(0); vv = np.zeros(0)
    for i in range(numPts-1):
        [found,t,val] = newtons(coeff[4*i],coeff[4*i+1],coeff[4*i+2],\
                                 coeff[4*i+3],time[i],time[i+1],tol,maxTries)
        if found:
            if (not localMaxOnly or \
               (2*coeff[4*i+2]+6*coeff[4*i+3]*(time[i]+time[i])//2 < 0) ):
                tt = np.append(tt,t)
                vv = np.append(vv,val)
        else:
            exitTimes += val
         
    print("Exitted domain {} of the {} times".format(exitTimes, len(time)))
    return [tt,vv]

def removeImg(t,v,tol):
    for i in range(len(t),0,-1):
        if t[i-1] > tol:
            continue
        else:
            return [t[:i],v[:i]]
    return [np.zeros(0),np.zeros(0)]

# https://www.geeksforgeeks.org/merge-sort/
def mergeSort(arr,idx,left,right): 
    length = right - left
    
    if length > 1: 
        mid = (left+right)//2 #Finding the mid of the array 

        mergeSort(arr,idx,left,mid) # Sorting the first half 
        mergeSort(arr,idx,mid,right) # Sorting the second half 
  
        i = j = k = 0
        arr2 = np.zeros(length)
        idx2 = np.zeros(length)
          
        # Copy data to temp arrays L[] and R[] 
        while i < length//2 and j < length-length//2: 
            if arr[i+left] < arr[j+mid]: 
                arr2[k] = arr[i+left] 
                idx2[k] = idx[i+left]
                i+=1
            else: 
                arr2[k] = arr[j+mid] 
                idx2[k] = idx[j+mid]
                j+=1
            k+=1
          
        # Checking if any element was left 
        while i < length//2: 
            arr2[k] = arr[i+left] 
            idx2[k] = idx[i+left]
            i+=1; k+=1
          
        while j < length-length//2: 
            arr2[k] = arr[j+mid] 
            idx2[k] = idx[j+mid]
            j+=1; k+=1
            
        arr[left:right] = arr2
        idx[left:right] = idx2

### random
def truncateToNumpy(v):
    return np.array(v)[:2**(round(log(len(v))/log(2)))]

def derDisrectize(v):
    d1 = np.array([v[i+1]-v[i] for i in range(len(v)-1)])
    return np.append(d1[0], d1)

def avg(v):
    return (1. * sum(v))/len(v)

def sma(v, length=10):
    smav = np.zeros(0)
    for i in range(len(v)):
        smav = np.append(smav, avg(v[max(0,i-length):i+1]))
    return smav