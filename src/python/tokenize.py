#!/usr/bin/python 

### TODO
# 1. Phrases
# 2. Fix names

### Usage
# Just make dict: no --dict and no --mode
# Make dict too: no --dict
# --mode=filelist : tokenize each file listed in argument file
# --mode=bigfile : treat arg file as one big document
# --mode=lines: treat each line as a separate document
# in cases where there is no --dict, supply --stopwords and --threshold; can supply it to further remove stopwords although they will eat up dict id space if you elave it in the dict


# For 1 field, just run this
# for multi field, create a .sh files that runs this repeatedly and paste

import sys
import codecs
import re
import getopt
from BeautifulSoup import BeautifulSoup

class Dictionary:
  def __init__(self, filename=None):
    self.freq = {}
    self.id2word = {}
    self.word2id = {}
    self.nextid = 0
    self.id2pos = {}
    self.pos_flag = 0
    if filename is not None:
      self.ReadIn(filename)
    self.stopwords = {}

  def ReadStopwords(self, filename):
    f = codecs.open(filename, encoding='utf-8')
    for word in f:
      word = word.rstrip()
      self.stopwords[word] = 1
    f.close()

  def ReadIn(self, filename):
    f = codecs.open(filename, encoding='utf-8')
    total_words = 0
    for line in f:
      line = line.rstrip()
      components = line.split("\t")
      id        = int(components[0])
      word      = components[1]
      if len(components) > 2:
        frequency = int(components[2])
      else:
        frequency = 0
      if len(components) > 3:
        pos = components[3]
        self.pos_flag = 1
        self.id2pos[id] = pos
      else:
        self.id2pos[id] = ''

      self.id2word[id]   = word
      self.word2id[word] = id
      self.freq[word] = frequency
      if id >= self.nextid:
        self.nextid = id + 1
      total_words += frequency
    f.close()
    sys.stdout.write(str(total_words) + " words in corpus\n")

  def Dump(self, filename):
    f = codecs.open(filename, encoding='utf-8', mode='w+')
    total_words = 0
    for id, value in self.id2word.iteritems():
      f.write(str(id) + '\t' + value + '\t' + str(self.freq[self.id2word[id]]) + '\n')
      total_words += self.freq[self.id2word[id]]
    f.close()
    sys.stdout.write(str(total_words) + " words in corpus\n")

  def Add(self, words):
    for word in words:
      if word not in self.word2id:
        self.id2word[self.nextid] = word
        self.word2id[word] = self.nextid
        self.nextid = self.nextid + 1
      self.freq[word] = self.freq.get(word, 0) + 1

  def Threshold(self, threshold):
    new_word2id = {}
    new_id2word = {}
    new_freq    = {}

    new_id = 0
    for word in self.word2id.keys():
      if self.freq[word] > threshold:
        new_freq[word] = self.freq[word]
        new_id2word[new_id] = word
        new_word2id[word]   = new_id
        new_id = new_id + 1

    self.freq    = new_freq
    self.id2word = new_id2word
    self.word2id = new_word2id

  def RemoveStopwords(self):
    new_word2id = {}
    new_id2word = {}
    new_freq    = {}

    new_id = 0
    for word in self.word2id.keys():
      if word not in self.stopwords:
        new_freq[word] = self.freq[word]
        new_id2word[new_id] = word
        new_word2id[word]   = new_id
        new_id = new_id + 1

    self.freq    = new_freq
    self.id2word = new_id2word
    self.word2id = new_word2id


class Tokenizer:
  def ReadStopwords(self, filename):
    f = codecs.open(filename, encoding='utf-8')
    for word in f:
      word = word.rstrip()
      self.stopwords[word] = 1
    f.close()

  def __init__(self, threshold, stopwords, process):
    self.threshold = threshold
    self.process   = process
    self.stopwords = {}
    if stopwords != None:
      self.ReadStopwords(stopwords)

  def CleanWord(self, word):
    word = re.sub('^[^\w]+', '', word)
    word = re.sub('[^\w]+$', '', word)
    word = word.lower()
    return word

  def Tokenize(self, line):
    sel_words = []
    line = unicode(BeautifulSoup(line, convertEntities=BeautifulSoup.HTML_ENTITIES))
    if self.process:
#      words = re.split('[\\\\\s\-\/\(\);\'\]\[,:\.\>\<]+', line)
#      orig_words = re.split('([\\\\\s\-\/\(\);\'\]\[,:\.\>\<]+)', line)
      words = re.compile('\W+', re.U).split(line)
      orig_words = re.compile('(\W+)', re.U).split(line)
    else:
      words = re.split('\s+', line)
      orig_words = re.split('(\s+)', line)
    for word in words:
      word = self.CleanWord(word)
      if len(word) < 1:
        continue
      if self.stopwords.get(word, 0) != 0:
        continue
      if re.search('[A-Za-z]', word) == None:
        continue
      sel_words.append(word)
    return sel_words, orig_words

  def BOWLine(self, dictionary, line):
    bow = {}
    tot_words = 0
    words, orig_words = self.Tokenize(line)
#    sys.stdout.write("corpus line count: " + str(len(words)) + "\n")
#    sys.stdout.write("corpus line : "  + line)
#    sys.stdout.write("corpus words : " + str(words))
    for word in words:
      if word in dictionary.word2id:
        bow[dictionary.word2id[word]] = bow.get(dictionary.word2id[word], 0) + 1
        tot_words = tot_words + 1
#sys.stdout.write("bow count: " + str(tot_words) + "\n")

    o_cnt = 0 
    for i in range(len(orig_words)):
      o_word = orig_words[i]
      clean_word = self.CleanWord(o_word)
      if clean_word in dictionary.word2id:
        orig_words[i] = o_word + '' + str(dictionary.word2id[clean_word])
        o_cnt = o_cnt + 1
    if o_cnt != tot_words:
      print "Whaa paint word fail"
    return bow, tot_words, orig_words

  def Stringize(self, bow, tot_words):
    o_str = str(tot_words) + ' '
    for id in sorted(bow.keys()):
      num = bow[id]
      o_str = o_str + ((str(id) + ' ') * num)
    return o_str + '\n'

  def Paint(self, orig_words):
    return u''.join(orig_words) + '\n'

  def BOWFile(self, dictionary, file):
    f = codecs.open(file, encoding='utf-8')
    if file == '-':
      f = codecs.getreader('utf-8')(sys.stdin)
    big_line = ''
    for line in f:
      line = line.rstrip()
      big_line = big_line + line + ' '
    f.close()
    bow, tot_words, orig_words = self.BOWLine(dictionary, big_line)
    return bow, tot_words, orig_words

  def ConvertLines(self, dictionary, corpus, bow_file, paint_file):
    f = codecs.open(corpus, encoding='utf-8')
    for line in f:
      line = line.rstrip()
      bow, tot_words, orig_words = self.BOWLine(dictionary, line)
      bow_file.write(self.Stringize(bow, tot_words))
      if paint_file != None:
        paint_file.write(self.Paint(orig_words))
    f.close()

  def ConvertFile(self, dictionary, corpus, bow_file, paint_file):
    bow, tot_words, orig_words = self.BOWFile(dictionary, corpus)
    bow_file.write(self.Stringize(bow, tot_words))
    if paint_file != None:
      paint_file.write(self.Paint(orig_words))
    return tot_words
      
  def MassConvert(self, dictionary, corpus, bow_file, paint_file):
    f = codecs.open(corpus, encoding='utf-8')
    ctr = 0
    tot_words = 0
    for line in f:
      line = line.rstrip()
      tot_words += self.ConvertFile(dictionary, line, bow_file, paint_file)
      ctr += 1
      if ctr == 100:
        sys.stdout.write('.')
        ctr = 0
    f.close()
    sys.stdout.write(str(tot_words) + " words in corpus\n")

  def AddToDict(self, dictionary, filename):
    f = codecs.open(filename, encoding='utf-8')
    tot_words = 0
    for line in f:
      line = line.rstrip()
      words, orig_words = self.Tokenize(line)
      dictionary.Add(words)
      tot_words += len(words)
    f.close()
#    sys.stdout.write("dict line count: " + str(tot_words) + "\n")

  def MakeDict(self, filename, mode):
    dictionary = Dictionary()
    if mode == 'filelist':
      f = codecs.open(filename, encoding='utf-8')
      ctr = 0
      for line in f:
        line = line.rstrip()
        self.AddToDict(dictionary, line)
        ctr += 1
        if ctr == 100:
          sys.stdout.write('.')
          ctr = 0
      f.close()
    else:
      self.AddToDict(dictionary, filename)
    dictionary.Threshold(self.threshold)
    return dictionary

def main():
  opts, args = getopt.getopt(sys.argv[1:], 'x', ['stopwords=', 'noprocess', 'threshold=', 'dict=', 'mode=', 'paintfile=', 'dictmode='])

  stopwords = None
  process   = 1
  threshold = 0

  dict_file = ''
  mode = ''
  dictmode = 'onefile'

  paint_file_name = ''

  for o, v in opts:
    if o == '--stopwords':
      stopwords = v
    elif o == '--noprocess':
      process = 0
    elif o == '--threshold':
      threshold = int(v)
    elif o == '--dict':
      dict_file = v
    elif o == '--mode':
      mode = v
    elif o == '--paintfile':
      paint_file_name = v
    elif o == '--dictmode':
      dictmode = v

  corpus        = args[0]
  bow_file_name = args[1]

  bow_file   = open(bow_file_name, 'w');
  paint_file = None
  if (paint_file_name != ''):
    paint_file = codecs.open(paint_file_name, 'w', encoding='utf-8');

  tokenizer = Tokenizer(threshold, stopwords, process)
  dictionary = Dictionary()
  if dict_file == '':
    output_dict = args[2]
    dictionary  = tokenizer.MakeDict(corpus, dictmode)
    dictionary.Dump(output_dict)
    sys.stdout.write('\n')
  else:
    dictionary.ReadIn(dict_file)

  if mode != '':
    if mode == 'bigfile':
      tokenizer.ConvertFile(dictionary, corpus, bow_file, paint_file)
    elif mode == 'filelist':
      print "File list mode\n"
      tokenizer.MassConvert(dictionary, corpus, bow_file, paint_file)
    elif mode == 'lines':
      tokenizer.ConvertLines(dictionary, corpus, bow_file, paint_file)

    bow_file.close()
    if (paint_file_name != ''):
      paint_file.close()

if __name__ == "__main__":
  main()

sys.stdout.write('\n')
