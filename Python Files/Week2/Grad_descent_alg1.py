# Implement gradient descent method to find the root of x^2-2=0
# we want to find the root of f(x) = x^2-2

import time
import matplotlib.pyplot as plt
import numpy as np

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

## Define residual function
prev_res = x**2 - 2

dres = 1 # Change in value of residual function

## Number of iterations
i = 0

## loop till change in residual function < error value
while dres>err:
    #### Update iteration number
    i += 1

    #### update the input of function by going opposite to steepest gradient
    x = x - alpha*(2*x) # x(k+1) = x(k) - alpha*f'(x)

    #### Update value of residual function
    res = x**2 - 2

    #### Change in residual function
    dres = abs(res - prev_res)

    #### set previous residual to be new now calculated one
    prev_res = res

    #### plot current location of our point
    plt.scatter(x,res,s=15)
    
## End the timer
dt = time.time() - dt

## Output
#### Print value of x for which f(x) is minimum
print(f'x at which f(x) is minimum is {x}')

#### Print the minimum value of f(x)
print(f'Minimum value of f(x) is {res}')

#### Print the change in f(x) and corresponding change in x when iteration stops
print(f'Change in f(x) is {dres} and the corresponding change in x is {-alpha*2*x} for which the iteration stops')

#### Print number of iterations required
print(f'Number of iterations happened are {i}')

#### Time required for the function to run
print(f'Time required to complete the iterations is {dt}')

#### Plot f(x) = x**2-2
x = np.linspace(-0.5,2,100)
y = x**2-2
plt.plot(x,y)

#### Show the plot
plt.show()
