import tokenize
import sys

dictionary = tokenize.Dictionary(sys.argv[1])
tokenizer = tokenize.Tokenizer(0, None, 0)

bow_neg, tot_words, orig_words = tokenizer.BOWFile(dictionary, sys.argv[2])
bow_pos, tot_words, orig_words = tokenizer.BOWFile(dictionary, sys.argv[3])


for word in bow_neg.keys():
  print "0 " + str(word) + " 0"

for word in bow_pos.keys():
  print "0 " + str(word) + " 1"
