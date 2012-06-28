import sys
for line in sys.stdin:
  a = line.split('\t')
  print float(a[0]) - float(a[1])
