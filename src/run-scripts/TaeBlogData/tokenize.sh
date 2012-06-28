Do () {
  python/tokenize.py --paintfile TaeBlogDataProject/$1.paintfile \
                     --mode filelist \
                     --stopwords LeePangProject/stopwords.txt \
                     --threshold $2 \
                     --dictmode filelist \
                     TaeBlogDataProject/$1.filelist \
                     TaeBlogDataProject/$1.bow \
                     TaeBlogDataProject/$1.dict.words
}

Tokenize () {
  thresh_RS=5
  thresh_DK=10
  for b in RS DK
  do
    for p in bodies cmnts
    do
      z=thresh_$b
      Do $b.$p ${!z}
    done
  done
  Do Combo.bodies 10
}

MakeSeeds() {
  for b in RS DK
  do
    for i in positive negative
    do
      sort -k2,2 TaeBlogDataProject/$b.cmnts.dict.words | join -o 1.1 -1 2 "-" TaeBlogDataProject/$i.seeds > /tmp/$i
    done
    cat /tmp/positive  | perl -lne 'chomp; print "0 $_ 0"' > TaeBlogDataProject/$b.cmnts.seed_labels
    cat /tmp/negative  | perl -lne 'chomp; print "0 $_ 1"' >> TaeBlogDataProject/$b.cmnts.seed_labels
  done
}

#Tokenize
#MakeSeeds

Do RS.cmnts 5
#Do RS.bodies .html_ents
