import sys
import re

f = open(sys.argv[1], 'r')

poss = {}
for line in f:
  line = line.rstrip()
  words = re.split('[\s]+', line)
  for word in words:
    parts = word.split('_')
    if len(parts) < 2:
      continue
    tok = parts[0]
    pos = parts[1]
    
    if not tok in poss:
      poss[tok] = {}
    poss[tok][pos] = poss[tok].get(pos, 0) + 1

for w in poss:
  tags = poss[w].keys()
  tags.sort(key = lambda x:poss[w][x], reverse=True)
  sys.stdout.write(w + '\t')
#  for tag in tags:
#    sys.stdout.write(tag + ':' + str(poss[w][tag]) + ',')
  sys.stdout.write(tags[0] + ':' + str(poss[w][tags[0]]))
  sys.stdout.write('\n')
