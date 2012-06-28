import sys
import re

for line in sys.stdin:
  ids = re.split('[\s]+', line)
  ids.pop()
  ids.pop(0)
  target = ids.pop()
  fv = {}
  for t in ids:
    j = int(t) + 1
    if j not in fv:
      fv[j] = 0
    fv[j] += 1
  sys.stdout.write(str(target) + " ")
  for x in sorted(fv.keys()):
    y = fv[x]
    sys.stdout.write(str(x) + ":" + str(y) + " ")
  sys.stdout.write("\n")

