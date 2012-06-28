import sys
for line in sys.stdin:
  line = line.rstrip()
  a = line.split(',')
  print "2 %d %d" % (int(a[0]) - 1, int(a[1]) - 1)
