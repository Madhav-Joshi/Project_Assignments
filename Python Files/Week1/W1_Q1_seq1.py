x1 = input('Provide a non zero initial number:')

try:
    x1 = float(x1)
    var = 1/x1
except :
    quit('Please provide a non zero decimal number!')

e0 = 0.0001
change = abs((x1 + (2/x1))/2 - x1)

while change>e0:
    x1 = (x1 + (2/x1))/2
    change = abs((x1 + (2/x1))/2 - x1)

print(x1)