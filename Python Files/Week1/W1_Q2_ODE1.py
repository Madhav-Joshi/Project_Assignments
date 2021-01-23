import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x,a,b,c,d):
    return b*np.sin(a*x+d)+c*x

a,b = 0,1
ua,ub = 0,0
n = 20

h = (b-a)/(n+1)
x = np.linspace(a,b,n+2)[1:-1]

# B = -f(x) should work on entire array x
B = -np.sin(x)
# Changing first and last term of b
B[0] -= ua/h**2
B[-1] -= ub/h**2

# Making nxn array A.
A = np.identity(n)*(-2/h**2)
d0 = np.identity(n-1)/h**2
# Method 1
r,c = 0,1
A[r:r+d0.shape[0], c:c+d0.shape[1]] += d0
r,c = 1,0 
A[r:r+d0.shape[0], c:c+d0.shape[1]] += d0
# Method 2
'''d1 = np.pad(d0,((1,0),(0,1)),mode = 'constant',constant_values = (0,0))
d2 = np.pad(d0,((0,1),(1,0)),mode = 'constant',constant_values = (0,0))
A += d1 + d2'''

u = np.linalg.solve(A,B)

plt.scatter(x,u)

# Curve Fitting
# pylint: disable=unbalanced-tuple-unpacking
popt, pcov = curve_fit(func,x,u,[6.7,1,-0.8,0])
temp_x = np.linspace(a,b,100)
curvefit = func(temp_x,*popt)
print(popt)

plt.plot(temp_x,curvefit)

plt.show()