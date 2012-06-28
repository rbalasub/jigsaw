import sys
import math

f = open (sys.argv[1], 'r')
topics = []

for line in f:
  line = line.rstrip("\r\n")
  node_probs = line.split('\t')
  for t in range(len(node_probs)):
    if t >= len(topics):   
      topics.append([])
    topics[t].append(float(node_probs[t]))

f.close()

f = open (sys.argv[2], 'w')
for a in topics:
  for p in [x / sum(a) for x in a]:
    f.write(str(p) + " ")
  f.write("\n")
