import sys

domain = 0
for line in sys.stdin:
  line = line.rstrip()
  num, name = line.split(' ')
  for i in range(int(num)):
    print domain
  domain += 1
