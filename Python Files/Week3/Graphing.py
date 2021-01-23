import numpy as np
import matplotlib.pyplot as plt

dt_arr,err_fe,err_be = np.loadtxt('euler_methods.dat',delimiter='\t',usecols=(0,1,2),skiprows=0,unpack=True,max_rows=5)

err_fe *= -1

fig,(ax,ax2) = plt.subplots(1,2)
ax.plot(dt_arr,err_fe,label = '-Forward Euler error')
ax.plot(dt_arr,err_be,label = 'Backward Euler error')
#ax.plot(dt_arr,0*dt_arr,color = 'black')
ax.set_xlim(dt_arr[0],dt_arr[-1])
ax.legend()
ax.grid(which='both')
ax.set(xscale = 'log', yscale='log',xlabel=r'Decreasing $\Delta t$ values',ylabel = '|Calculated value - True Value|', title = 'Error in Euler method values at t = 5 sec')

ax2.plot(dt_arr,err_be-err_fe,label = 'Difference in error values')
ax2.set(xlim = (dt_arr[0],dt_arr[-1]), xscale = 'log', yscale='log', xlabel = r'Decreasing $\Delta t$ values', ylabel = r'$\Delta Error$', title = r'$Error_{Back Euler} - Error_{Forward Euler}$')
#ax2.legend()
ax2.grid(which = 'both')

fig.suptitle(r"Euler Methods' comparison for $\frac{dx}{dt}=2x$")

plt.show()