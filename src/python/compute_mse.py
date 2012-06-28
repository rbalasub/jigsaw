import sys
n = 0
mse = 0.0
for line in sys.stdin:
  a,b = line.split('\t')
  a = float(a)
  b = float(b)
  
  mse += (a-b) * (a-b)
  n += 1

print mse / n
