import numpy as np 
from numpy import array as arr 

def A(p,d):
    # p = vector of size n, d = size of A
    return np.identity(d) - (p @ p.T)

def b(p,d):
    # p = vector of size n, d = size of A
    return np.ones((d,1))

def dfdx(x):
    return 2*x.T

def dbdp(p,d):
    n = len(p)
    return np.zeros((n,d,1))

def dAdp(p,d):
    n = len(p)
    dpA = np.zeros((n,d,d))
    for i in range(d):
        dpA[i,:,i] += p.reshape(dpA[i,:,i].shape)
        dpA[i,i,:] += p.reshape(dpA[i,i,:].shape)
    return dpA

def f(x):
    return np.linalg.norm(x)**2

eta = 0.1 # Gradient descent factor
dpg = 1 # initialize with arbitrary large value
iter_count = 0 # Number of iterations

n = 5 # size of p
d = n # size of x same as that of p for this problem

#p = np.random.rand(n,1) # Guess p (nx1)
p = np.ones((n,1)) 

# Find dg/dp
while np.linalg.norm(dpg)>0.001:
    iter_count+=1 # Update counter

    # Define A(p), b(p)
    Ap = A(p,d) # (dxd)
    bp = b(p,d) # (dx1)

    A_inv = np.linalg.inv(Ap) # Find inverse of A (dxd)
    x = A_inv @ bp # Find x (dx1)

    dxf = dfdx(x) # Partial of f(x) wrt x (1xd)
    dpb = dbdp(p,d) # Partial of b wrt p (nxdx1)

    # Making Partial of A wrt p i.e. find in terms of variable on paper then make it here
    dpA = dAdp(p,d) # (nxdxd)
        
    # Partial of x wrt p as [A^-1 @ (dpb - dpa @ x)]
    dpx = A_inv @ (dpb-(dpA @ x)) # (nxdx1)

    # Partial of g wrt p as partial of f wrt x times partial of x wrt p i.e. chain rule
    dpg = dxf @ dpx # (nx1x1)

    # Update p
    p = p + eta*dpg.reshape(p.shape)
    
print(f'Number of iterations = {iter_count}')
print(f'Value of function f(x) = {round(f(x),5)}')
#print(f'Value of p = {p}')
#print(f'Value of x = {x}')