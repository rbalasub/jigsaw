#!/usr/bin/python 
import sys
import tokenize
import codecs
import re

tokenizer = tokenize.Tokenizer(0, None, 0)
dictionary = tokenize.Dictionary()
dictionary.ReadStopwords(sys.argv[4])
f = codecs.open(sys.argv[1], encoding='utf-8')
i = 0
for line in f:
  t = 0
  line = line.rstrip()
  features = line.split(' ')
  words = []
  label_feat = features.pop()
  counts = {}
  for feat in features:
    components = feat.split(':')
    if re.search('_', components[0]):
      continue
    toks = components[0].split('_')
    for w in toks:
      word = tokenizer.CleanWord(w)
      if len(word) < 1:
        continue
      if dictionary.stopwords.get(word, 0) != 0:
        continue
      if re.search('[A-Za-z]', word) == None:
        continue
      words.append(word)
      counts[word] = int(components[1]) - 1
  dictionary.Add(words)
  for word in words:
    t += 1 + counts[word]
    dictionary.freq[word] = dictionary.freq.get(word, 0) + counts[word]
  i += 1
#  print t
  if i % 1000 == 0:
    sys.stdout.write('.') 
f.close()

print len(dictionary.word2id.keys()), " words"
dictionary.Threshold(5)
print len(dictionary.word2id.keys()), " words"
dictionary.RemoveStopwords()
dictionary.Dump(sys.argv[2])

f = codecs.open(sys.argv[1], encoding='utf-8')
of = codecs.open(sys.argv[3], 'w', encoding='utf-8')
for line in f:
  line = line.rstrip()
  features = line.split(' ')
  label_feat = features.pop()
  label_comps = label_feat.split(':')
  ids = []
  for feat in features:
    components = feat.split(':')
    if re.search('_', components[0]):
      continue
    toks = components[0].split('_')
    for w in toks:
      word = tokenizer.CleanWord(w)
      if word in dictionary.word2id:
        for i in range(int(components[1])):
          ids.append(dictionary.word2id[word])
  of.write(str(len(ids)) + ' ')
  for i in ids:
    of.write(str(i) + ' ')
  of.write(label_comps[1] + '\n')
f.close()
of.close()
