SLDA () {
  bin/link_lda --train_file MarkDredzeProject/$1.bow \
               --test_file  MarkDredzeProject/$1.test.bow \
               --output_prefix MarkDredzeProject/Output/$1 \
               --topics 10 \
               --num_real_targets 1 \
               --model_targets | tee MarkDredzeProject/$1.log
}

SSLDA () {
  bin/link_lda --train_file MarkDredzeProject/$1.bow \
               --output_prefix MarkDredzeProject/Output/$1 \
               --topics 10 \
               --num_real_targets 1 \
               --model_targets | tee MarkDredzeProject/$1.log
}

SSLDALabels () {
  bin/link_lda --train_file MarkDredzeProject/$1.bow \
               --output_prefix MarkDredzeProject/Output/$1.with_labels \
               --node_label_file MarkDredzeProject/$2 \
               --topics 10 \
               --num_real_targets 1 \
               --label_randomness 0.05 \
               --clamp_rigidity 0.95 \
               --theta_penalty 1  \
               --theta_variance 0.2 \
               --model_targets | tee MarkDredzeProject/$1.log
}

View () {
  python/view_model.py    --report=MarkDredzeProject/Output/$1.report  \
                          --entity_names=words \
                          --dict_prefix=MarkDredzeProject/$2 \
                          --output_prefix=MarkDredzeProject/HTML/$1 \
                          --index  \
                          --topics \
                          --title "Combo bodies two target" \
                          --target_names "Stars" 
}

for frac in 1.0 0.8 0.6 0.4 0.2
do
  for b in books dvd electronics kitchen
  do
    SLDA        $b.$frac.slda 
    View        $b.$frac.slda $b.dict.

    SSLDA       $b.$frac.s-slda 
    View        $b.$frac.s-slda $b.dict.

    SSLDALabels $b.$frac.s-slda $b.seed_labels
    View        $b.$frac.s-slda.with_labels $b.dict.
  done
done
