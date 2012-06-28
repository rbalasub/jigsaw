import sys
import re

for line in sys.stdin:
  a = re.split('[\s]+', line)
  a.pop()
  i = int(sys.argv[1]) + 1
  for item in a:
    sys.stdout.write(str(i) + ':' + item + ' ')
    i = i + 1
  print
    

