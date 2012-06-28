for b in RS DK
do
  # run on comments with seeds
  bin/link_lda --train_file TaeBlogDataProject/$b.cmnts.bow  \
               --node_label_file TaeBlogDataProject/$b.cmnts.seed_labels \
               --output_prefix TaeBlogDataProject/Output/$b.cmnts \
               --topics 10 \
               --label_randomness 0.05 \
               --clamp_rigidity 0.95 \
               --theta_penalty 1  \
               --theta_variance 0.2 | tee TaeBlogDataProject/Output/$b.cmnts.log

   python/view_model.py --report=TaeBlogDataProject/Output/$b.cmnts.report  \
                        --entity_names=words \
                        --dict_prefix=TaeBlogDataProject/$b.cmnts.dict. \
                        --output_prefix=TaeBlogDataProject/HTML/${b}_comments \
                        --index \
                        --topdocs \
                        --topics \
                        --docs_file TaeBlogDataProject/$b.cmnts.paintfile \
                        --docs_mode=paintlist \
                        --limit 100 \
                        --title "$b comments" \
                        --per_page 10

  # use comment membership in pos and neg topics as targets for bodies
  python/get_diff_topic_1_2.py < TaeBlogDataProject/Output/$b.cmnts.run.0.train.doc_topic_distr > TaeBlogDataProject/Output/$b.cmnts.target
  paste TaeBlogDataProject/$b.bodies.bow TaeBlogDataProject/Output/$b.cmnts.target > TaeBlogDataProject/$b.bodies.one_target.bow
 
  # train lda for bodies
  bin/link_lda --train_file TaeBlogDataProject/$b.bodies.one_target.bow \
               --output_prefix TaeBlogDataProject/Output/$b.bodies \
               --topics 10 \
               --num_real_targets 1 \
               --model_targets | tee TaeBlogDataProject/Output/$b.bodies.log

  python/view_model.py --report=TaeBlogDataProject/Output/$b.bodies.report  \
                        --entity_names=words \
                        --dict_prefix=TaeBlogDataProject/$b.bodies.dict. \
                        --output_prefix=TaeBlogDataProject/HTML/${b}_bodies \
                        --index \
                        --topdocs \
                        --topics \
                        --docs_file TaeBlogDataProject/$i.bodies.paintfile \
                        --docs_mode=paintlist \
                        --limit 100 \
                        --title "$b bodies one target" \
                        --target_names "Cmnt senti" \
                        --per_page 10

done

paste TaeBlogDataProject/Output/RS.cmnts.target TaeBlogDataProject/RS.false_target >  TaeBlogDataProject/Output/Combo.cmnts.target
paste TaeBlogDataProject/DK.false_target TaeBlogDataProject/Output/DK.cmnts.target >> TaeBlogDataProject/Output/Combo.cmnts.target
paste Combo.bodies.bow TaeBlogDataProject/Output/Combo.cmnts.target > TaeBlogDataProject/Combo.bodies.one_target.bow

bin/link_lda   --train_file TaeBlogDataProject/Combo.bodies.one_target.bow \
               --output_prefix TaeBlogDataProject/Output/Combo.bodies \
               --topics 8 \
               --num_real_targets 2 \
               --model_targets | tee TaeBlogDataProject/Output/Combo.bodies.log

python/view_model.py    --report=TaeBlogDataProject/Output/Combo.bodies.report  \
                        --entity_names=words \
                        --dict_prefix=TaeBlogDataProject/Combo.bodies.dict. \
                        --output_prefix=TaeBlogDataProject/HTML/Combo_bodies \
                        --index \
                        --topdocs \
                        --topics \
                        --docs_file TaeBlogDataProject/Combo.bodies.paintfile \
                        --docs_mode=paintlist \
                        --limit 100 \
                        --title "Combo bodies two target" \
                        --target_names "RS cmnt senti,DK cmnt senti" \
                        --per_page 10
