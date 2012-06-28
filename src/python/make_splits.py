#!/usr/bin/python 
import codecs
import random
import sys

for frac in [0.2, 0.4, 0.6, 0.8]:
  f  = codecs.open(sys.argv[1] + '.1.0.slda.bow', encoding='utf-8')
  of = codecs.open(sys.argv[1] + '.' + str(frac) + '.' + sys.argv[2] + '.bow', 'w', encoding='utf-8')
  if sys.argv[2] == 'slda':
    of2 = codecs.open(sys.argv[1] + '.' + str(frac) + '.' + sys.argv[2] + '.test.bow', 'w', encoding='utf-8')
  for line in f:
    if random.random() < frac:
      of.write(line)
    else:
      if sys.argv[2] == 'slda':
        of2.write(line)
      else:
        line = line.rstrip()
        words = line.split(' ')
        words.pop()
        words.append("NA")
        of.write(' '.join(words) + '\n')
  f.close()
  of.close()
  if sys.argv[2] == 'slda':
    of2.close()

