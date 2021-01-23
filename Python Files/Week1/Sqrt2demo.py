# Program to compute sqrt 2 using following iterative alg
#
# x_(n+1) = (1/2)(x_n + 2/x_n)

import time
import numpy as np
import matplotlib.pyplot as plt 

## Initialize a guess
x0 = 2

## Initialize timer
dt = time.time()

## Compute sqrt 2 using the iterative algorithm
#### Initialize error
err = 1

#### Initialize tolerance
tol = 1e-4

#### Initialize iteration counter
itr_count = 0

#### Open file to dump output
f_out = open('nr_iterations.dat','w')

###### Write initial conditions
f_out.write(f'{itr_count}\t{x0}\t{err}\n')

#### Loop until convergence (until error < tolerance)
while err>tol:
    ###### Update iteration counter
    itr_count += 1

    ###### Perform one step of iteration 
    x = 0.5*(x0 + 2.0/x0)

    ###### Compute error
    err = abs(x**2-2)

    ###### Copy previous iterate to current value
    x0 = x

    ###### Print status of iteration
    print(f'Iteration {itr_count}: x = {x}, error = {err}')

    ###### Print status of iteration to file
    f_out.write(f'{itr_count}\t{x}\t{err}\n')

#### Close output file
f_out.close()

#### Assign vectors from the file made
vec_itr, vec_x, vec_err = np.loadtxt('nr_iterations.dat',delimiter='\t',usecols=(0,1,2),unpack=True)
print(vec_itr, vec_x)

## Finalise timer
dt = time.time() - dt

## Output
#### Plot solution
plt.plot(vec_itr,vec_x)
plt.show()
plt.plot(vec_itr, vec_err)
plt.show()

#### Print solution
print(f'Square root of 2 is {x}')

#### Print number of iterations to convergence
print(f'Number of iterations to convergence = {itr_count}')

#### Print time elapsed
print(f'Time elapsed = {dt} s')