# Implement gradient descent method to find the root of x^2-2=0
# we want to find the root of f(x) = x^2-2

import time
import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return x**2-2

def ddx(x):
    return 2*x

## Initiate a guess x0
x0 = 2

## Set iteration factor alpha
alpha = 0.1

## When change in f is less than this then stop the iteration
err = 0.001

## Start the time
dt = time.time()

## Initialize x to x0
x = x0

## Define residual function and iterate values
res = ddx(x)
ansx = np.array([x])
ansy = np.array([func(x)])

## Number of iterations
i = 0

## loop till change in residual function < error value
while res>err:
    #### Update iteration number
    i += 1

    #### update the input of function by going opposite to steepest gradient
    x = x - alpha*(ddx(x)) # x(k+1) = x(k) - alpha*f'(x)

    #### Update value of residual function
    res = ddx(x)

    #### plot current location of our point
    #plt.scatter(x,func(x),s=15)
    ansx = np.append(ansx,x)
    ansy = np.append(ansy,func(x))
    
## End the timer
dt = time.time() - dt

## Output
#### Print value of x for which f(x) is minimum
print(f'x at which f(x) is minimum is {x}')

#### Print the slope i.e. residual when iterations stops
print(f'Slope at which iteration stops is {res}')

#### Print number of iterations required
print(f'Number of iterations happened are {i}')

#### Time required for the function to run
print(f'Time required to complete the iterations is {dt}')

#### Plot the iteration values
plt.scatter(ansx,ansy,s=15,c = ansx,cmap='Wistia',label = 'Iteration value')
plt.colorbar()

#### Plot f(x) = x**2-2
x = np.linspace(-0.5,2,100)
y = func(x)
plt.plot(x,y, color = 'green', alpha = 0.5, label = '$f(x)=x^2-2$')
plt.legend()
plt.xlabel('X')
plt.ylabel('$f(x)$')
plt.title('Gradient Descent Algorithm')

#### Show the plot
plt.show()
