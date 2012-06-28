import sys

f = open(sys.argv[1], 'r')
g = open(sys.argv[2], 'w')

ctr = 0
for label in f:
  g.write("0 " + str(ctr) + " " + str(int(label) - 1) + "\n")
  ctr = ctr + 1
  
