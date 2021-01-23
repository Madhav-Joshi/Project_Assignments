# dx/dt = f(x)

# Forward euler is (x(t+ dt) - x(t))/dt = f(x(t))
# Backward euler is (x(t+ dt) - x(t))/dt = f(x(t + dt))
# In general Backward euler is better

# Solving for f(x) = 2x s.t. x(0)=1
# We know its solution as x = e**(2t)
# Solving the same by forward and backward euler

import numpy as np
import matplotlib.pyplot as plt

## Setting the initial conditions
ansf = np.array([1])
ansb = np.array([1])
dt = 0.001
x = 1
iter_count = int(5/dt)

## Forward Euler
#### Calculating further terms using x(t+ dt) = (1+ (dt)*2)*x(t) 
for i in range(iter_count):
    x = (1+2*dt)*x
    ansf = np.append(ansf,x)

## Backward Euler
#### Calculating further terms using x(t+ dt) = x(t)/(1-2*dt)
x = ansb[0]
for i in range(iter_count):
    x = x/(1-2*dt)
    ansb = np.append(ansb,x)

## Time array corresponding to iter_count and dt
t = np.linspace(0,dt*iter_count,len(ansf))

## Output
#### Initialize figure
fig = plt.figure()

#### Plot Forward Euler ansf
plt.plot(t,ansf, label = 'Forward Euler')

#### Plot Backward Euler ansb
plt.plot(t,ansb, label = 'Backward Euler')

#### Plot original solution
plt.plot(t,np.exp(2*t), label = 'Original solution')

plt.legend()
plt.xlabel('Time')
plt.ylabel('$x(t)$')
plt.title(r'Euler Methods ($\frac{dx}{dt} = 2x$)')

#plt.yscale('log')

plt.show()