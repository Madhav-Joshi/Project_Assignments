# Strating number
x = 2 
error = 0.001
#residual function
r = abs(x**2 - 2)
while r >= error:
    x = (x + (2/x))/2
    r = r = abs(x**2 - 2)
print(x) 
