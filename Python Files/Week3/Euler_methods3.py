# dx/dt = f(x)

# Forward euler is (x(t+ dt) - x(t))/dt = f(x(t))
# Backward euler is (x(t+ dt) - x(t))/dt = f(x(t + dt))
# In general Backward euler is better

# Solving for f(x) = 2x s.t. x(0)=1
# We know its solution as x = e**(2t)
# Solving the same by forward and backward euler

import numpy as np
import matplotlib.pyplot as plt

## Backward euler works only if func is of the form k*x where k is a constant
def func(x):
    return 2*x

def solution(time_arr):
    return np.exp(2*time_arr)

def fe(iter_count,x,dt):
    # Forward Euler
    ## Calculating further terms using x(t+ dt) = (1+2*dt)x(t)
    ansf = np.zeros(iter_count+1)
    ansf[0] = x
    for i in range(iter_count):
        x = x+func(x)*dt
        ansf[i+1] = x
    return ansf

def be(iter_count,x,dt):
    # Backward Euler
    ## Calculating further terms using x(t+ dt) = x(t)/(1-2*dt)
    ansb = np.zeros(iter_count+1)
    ansb[0] = x
    for i in range(iter_count):
        x = x/(1-func(x)*dt/x)
        ansb[i+1]= x
    return ansb

## Setting the initial conditions
dt = 0.1 # dt should divide t
t = 5 # time
x = 1 # Boundary condition x(t=0)=1
iter_count = int(t/dt)

err_no = 7
err_fe = np.zeros(err_no)
err_be = np.zeros(err_no)
dt_arr = np.zeros(err_no)
sol = solution(t) # solution at time t (= 5 in this case)

# Open file to dump output
f_out = open('euler_methods.dat','w')

for i in range(err_no):
    dt = dt/(2**i)
    dt_arr[i] = dt
    iter_count = int(t/dt)
    err_fe[i] = fe(iter_count,x,dt)[-1] - sol
    err_be[i] = be(iter_count,x,dt)[-1] - sol
    f_out.write(f'{dt}\t{err_fe[i]}\t{err_be[i]}\n')

# Close the file
f_out.close()

fig,ax = plt.subplots()
ax.plot(dt_arr,err_fe,label = 'Forward Euler error')
ax.plot(dt_arr,err_be,label = 'Backward Euler error')
ax.plot(dt_arr,0*dt_arr,color = 'black')
ax.set_xlim(dt_arr[0],dt_arr[-1])
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()

'''## Forward Euler
ansf = fe(iter_count,x,dt)

## Backward Euler
ansb = be(iter_count,x,dt)

## Time array corresponding to iter_count and dt
t_arr = np.linspace(0,t,len(ansf))

## Output
#### Initialize figure
fig = plt.figure()

#### Plot Forward Euler ansf
plt.plot(t_arr,ansf, label = 'Forward Euler')

#### Plot Backward Euler ansb
plt.plot(t_arr,ansb, label = 'Backward Euler')

#### Plot original solution
plt.plot(t_arr,solution(t_arr), label = 'Original solution')

plt.legend()
plt.xlabel('Time')
plt.ylabel('$x(t)$')
plt.title(r'Euler Methods ($\frac{dx}{dt} = 2x$)')

#plt.yscale('log')

plt.show()'''