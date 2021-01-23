import numpy as np
from mpl_toolkits import mplot3d 
import matplotlib.pyplot as plt
# I'm taking the example of u(x)=x**3+y**3
# Solving 2D ODE a*d2u/dx2 + b*d2u/dxdy + c*d2u/dy2 + f(x,y) = 0  
# Coeff of partial derivative terms
a,b,c = 1,1,1 # <-- Change values here

# Making the domain (Rectangular for now)
x1,x2 = -10,10 # <-- Change values here
y1,y2 = -15,15 # <-- Change values here
n1,n2 = 20,20 # <-- Change values here for number of values in x and y direction respectively
h1 = (x2-x1)/n1
h2 = (y2-y1)/n2

x_temp = np.linspace(x1,x2,n1+1)
y_temp = np.linspace(y1,y2,n2+1)
x,y = np.meshgrid(x_temp,y_temp)

fx = -6*(x+y) # <-- Enter function here

# Boundary Conditions
uy1 = np.array([x_temp])**3+y1**3 # u(x,y1) <-- Change values here
uy2 = np.array([x_temp])**3+y2**3 # u(x,y2) <-- Change values here
ux1 = np.array([y_temp])**3+x1**3 # u(x1,y) <-- Change values here
ux2 = np.array([y_temp])**3+x2**3 # u(x2,y) <-- Change values here

# Linearly approximated ODE terms coefficients
coef1 = a/h1**2 # u(i-1,j)
coef2 = c/h2**2 # u(i,j-1)
coef3 = -2*a/h1**2 + b/(h1*h2) -2*c/h2**2 # u(i,j)
coef4 = a/h1**2 - b/(h1*h2) # u(i+1,j)
coef5 = -b/(h1*h2) + c/h2**2 # u(i,j+1)
coef6 = b/(h1*h2) # u(i+1,j+1)

# For solving AU=B
# Making Matrix A of size ((n1-1)(n2-1))x((n1-1)(n2-1))
# A is made of (n2-1)x(n2-1) stacked matrices each of size (n1-1)x(n1-1)
# There are 3 types of (n1-1)x(n1-1) matrices
# Making diagonal matrices for making above
d1 = np.identity(n1-2)*coef1
d3 = np.identity(n1-1)*coef3
d4 = np.identity(n1-2)*coef4
d5 = np.identity(n1-1)*coef5
d6 = np.identity(n1-2)*coef6
d2 = np.identity(n1-1)*coef2

# Making the first (n1-1)x(n1-1) type 
M = d3
row,col = 0,1
M[row:row+d4.shape[0], col:col+d4.shape[1]] += d4
row,col = 1,0 
M[row:row+d1.shape[0], col:col+d1.shape[1]] += d1
M1 = M

# Making the second (n1-1)x(n1-1) type 
M = d5
row,col = 0,1
M[row:row+d6.shape[0], col:col+d6.shape[1]] += d6
M2 = M

# Making the third (n1-1)x(n1-1) type 
M3 = d2

# Making the matrix A
z = np.zeros(((n1-1)*(n2-1),(n1-1)*(n2-1)))
row,col = 0,0
z[row:row+M1.shape[0], col:col+M1.shape[1]] += M1
for i in range(1,n2-1):
    row,col = (n1-1)*i,(n1-1)*i
    z[row:row+M1.shape[0], col:col+M1.shape[1]] += M1
    row,col = (n1-1)*(i-1),(n1-1)*i
    z[row:row+M2.shape[0], col:col+M2.shape[1]] += M2
    row,col = (n1-1)*i,(n1-1)*(i-1)
    z[row:row+M3.shape[0], col:col+M3.shape[1]] += M3
A = z

# Making B for AU=B
b = -fx[1:-1,1:-1]
b[:1,:] += -coef2*uy1[:,1:-1] # First row
b[:,:1] += -coef1*np.transpose(ux1[:,1:-1]) # First column
b[:,-1:] += -coef4*np.transpose(ux2[:,1:-1]) # Last column
b[:-1,-1:] += -coef6*np.transpose(ux2[:,2:-1]) # Last column
b[-1:,:] += -coef5*uy2[:,1:-1] - coef6*uy2[:,2:] # Last row

B = np.ravel(b)

# Solving AU=B (linear equations)
u = np.linalg.solve(A,B)
U = np.reshape(u,(n1-1,n2-1))

#Plotting the figure
fig = plt.figure()
#We can add multiple subplots to the same plot.
ax = fig.add_subplot(111, projection='3d')

X = x[1:-1,1:-1]
Y = y[1:-1,1:-1]

#cmap is colour map
surf = ax.plot_surface(X, Y, U,cmap='viridis', edgecolor='none')
fig.colorbar(surf, shrink=0.4, aspect=7)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
'''# Creating figure
fig = plt.figure() #figsize = (16, 9)
ax = plt.axes(projection ="3d")

# Creating color map
my_cmap = plt.get_cmap('hsv')

sctt = ax.scatter3D(x[1:-1,1:-1], y[1:-1,1:-1], U,
                    alpha = 0.8,
                    cmap = my_cmap)

plt.title("simple 3D scatter plot")
ax.set_xlabel('X-axis', fontweight ='bold') 
ax.set_ylabel('Y-axis', fontweight ='bold') 
ax.set_zlabel('Z-axis', fontweight ='bold')
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
 
# show plot
plt.show()'''