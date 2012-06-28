MakeSeeds() {
  for b in  books dvd electronics kitchen
  do
    for i in positive negative
    do
      sort -k2,2 MarkDredzeProject/$b.dict.words | join -o 1.1 -1 2 "-" MarkDredzeProject/$i.seeds > /tmp/$i
    done
    cat /tmp/positive  | perl -lne 'chomp; print "0 $_ 0"' > MarkDredzeProject/$b.seed_labels
    cat /tmp/negative  | perl -lne 'chomp; print "0 $_ 1"' >> MarkDredzeProject/$b.seed_labels
  done
}

Tokenize() {
  for i in books dvd electronics kitchen
  do 
    echo $i
    python/convert_mark.py ~/Foundry47/DataStore/dredze_sentiment/processed_stars/$i/all_balanced.review \
                           MarkDredzeProject/$i.dict.words \
                           MarkDredzeProject/$i.1.0.slda.bow

  cp MarkDredzeProject/$i.1.0.slda.bow MarkDredzeProject/$i.1.0.s-slda.bow
  done

}

MakeSplits() {
  for i in books dvd electronics kitchen
  do 
    echo $i
    python/make_splits.py MarkDredzeProject/$i slda
    python/make_splits.py MarkDredzeProject/$i s-slda
  done
}

Tokenize
MakeSeeds
MakeSplits


