import sys
import random

for line in sys.stdin:
  if random.random() <= 0.2:
    sys.stdout.write(line)
  else:
    sys.stderr.write(line)
    
