# Solving d2u/dx2 + sinx = 0, u:[0,1]->R, u(0)=u'(1)=0
# By finite element method
import numpy as np
import matplotlib.pyplot as plt

def solution(n,dx,f,ua,u1b): # f is of size n+2
    # Solution of Ku = Mf system of linear equations
    ## Make K = | 2 -1  0  0  ...  0 |
    ##          |-1  2 -1  0  ...  0 |
    ##          | 0 -1  2  .  ...  0 | * 1/dx
    ##          | .        .   .   . |
    ##          | 0  ...   ... -1  1 |
    K = 2*np.identity(n)
    K[-1,-1] *= 0.5
    dk = -1*np.identity(n-1)
    row,col = 0,1
    K[row:row+dk.shape[0], col:col+dk.shape[1]] += dk
    row,col = 1,0 
    K[row:row+dk.shape[0], col:col+dk.shape[1]] += dk
    K *= 1/dx

    ## Make M
    M = 2*dx*np.identity(n)/3
    dm = dx*np.identity(n-1)/6
    row,col = 0,1
    M[row:row+dm.shape[0], col:col+dm.shape[1]] += dm
    row,col = 1,0 
    M[row:row+dm.shape[0], col:col+dm.shape[1]] += dm

    ## Solve Ku = Mf
    #### Subtract (-ua/dx) from first element of Mf and (-u'b/dx) from last element
    rhs = np.matmul(M,f[1:-1])
    rhs[0] += dx*f[0]/6 - (-ua/dx) # = M[1,0]*f[0] - K[1,0]*u[0]
    #rhs[-1]+= dx*f[-1]/6 - (-u1b/dx) # = M[n,n+1]*f[n+1] - K[n,n+1]*u[n+1] or = M[-2,-1]*f[-1] - K[-2,-1]*u[-1]
    #u1b 
    #### Solve u = K^-1*rhs
    u = np.matmul(np.linalg.inv(K),rhs)
    u = np.insert(u,[0,n],[ua,u[-1]])

    return u


a = 0
b = 1
ua = 0
u1b = 0
n1 = [2,4,10,20,100]

for n in n1:
    x = np.linspace(a,b,n+2)
    dx = (b-a)/(n+1) 
    f = np.sin(x)
    u = solution(n,dx,f,ua,u1b)
    plt.plot(x,u,label = f'n = {n}')

## Plot the solution
#plt.plot(x,u)
contx = np.linspace(0,1,101)
plt.plot(contx,np.sin(contx)-np.cos(1)*contx,label = 'Actual Solution')
plt.legend()
plt.title(r"1D Finite Element Method ($\frac{d^2u}{dx^2}+\sin(x)=0$, $u(0)=0$, $u'(1)=0$)")
plt.xlabel('X')
plt.ylabel('Y = u(x)')
plt.show()