labs[1]='--node_label_file projects/MultiDomainProject/uni.node_labels'
labs[0]=' '
for targets in NO YES
do
  for labels in 0 1
  do
    bin/link_lda --train_file projects/MultiDomainProject/uni.train.docs.domains      \
                 --topics 75                                                          \
                 --md_splits 15,15,15,15,15                                           \
                 --md_probs 0.8,0.17,0.03                                             \
                 ${labs[$labels]}                                                     \
                 --md_num_domains 4                                                   \
                 --md_theta_penalty 1                                                 \
                 --num_real_targets 1                                                 \
                 --model_targets $targets                                             \
                 --output_prefix projects/MultiDomainProject/Output/big-uni-labels-${labels}_targets-$targets

    python src/python/view_model.py --report=projects/MultiDomainProject/Output/big-uni-labels-${labels}_targets-$targets.report    \
                                    --entity_names=words                                     \
                                    --dict_prefix=projects/MultiDomainProject/uni.dict.      \
                                    --output_prefix=projects/MultiDomainProject/HTML/uni-labels-${labels}_targets-$targets      \
                                    --index                                                  \
                                    --topics                                                 \
                                    --title "Unigram Multi domain Labels $labels Targets $targets"   \
                                    --target_names "Stars"                                   \
                                    --no_sort_topics                                         \
                                    --domain_names books,dvd,electronics,kitchen             \
                                    --wordlist projects/MultiDomainProject/uni.node_labels   \
                                    --topicdistr

#  cat projects/MultiDomainProject/uni.train.docs.domains | cut -f2 | \
#    paste -d'' "-" projects/MultiDomainProject/Output/uni-labels-${labels}_targets-$targets.run.0.train.doc_topic_distr  | \
#    python src/python/combine_topic_distr_by_domain.py 30  \
#  > projects/MultiDomainProject/HTML/uni-labels-${labels}_targets-$targets.domain_distr
  done
done
