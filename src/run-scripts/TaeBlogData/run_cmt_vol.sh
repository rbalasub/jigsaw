for b in RS DK
do
  paste TaeBlogDataProject/$b.bodies.bow TaeBlogDataProject/$b.cmnts.volume.norm > TaeBlogDataProject/$b.bodies.volume_target.bow
 
  # train lda for bodies
  bin/link_lda --train_file TaeBlogDataProject/$b.bodies.volume_target.bow \
               --output_prefix TaeBlogDataProject/Output/$b.bodies.volume \
               --topics 10 \
               --num_real_targets 1 \
               --model_targets | tee TaeBlogDataProject/Output/$b.bodies.volume.log
               

  python/view_model.py --report=TaeBlogDataProject/Output/$b.bodies.volume.report  \
                        --entity_names=words \
                        --dict_prefix=TaeBlogDataProject/$b.bodies.dict. \
                        --output_prefix=TaeBlogDataProject/HTML/${b}_bodies_volume \
                        --index \
                        --topdocs \
                        --topics \
                        --docs_file TaeBlogDataProject/$b.bodies.paintfile \
                        --docs_mode=paintlist \
                        --limit 100 \
                        --title "$b bodies volume target" \
                        --target_names "Cmnt volume" \
                        --per_page 10

done

paste TaeBlogDataProject/Output/RS.cmnts.target \
      TaeBlogDataProject/RS.cmnts.volume.norm   \
      TaeBlogDataProject/RS.false_target        \
      TaeBlogDataProject/RS.false_target        >  TaeBlogDataProject/Output/Combo.cmnts.volume

paste TaeBlogDataProject/DK.false_target        \
      TaeBlogDataProject/DK.false_target        \
      TaeBlogDataProject/Output/DK.cmnts.target \
      TaeBlogDataProject/DK.cmnts.volume.norm   >> TaeBlogDataProject/Output/Combo.cmnts.volume
paste TaeBlogDataProject/Combo.bodies.bow TaeBlogDataProject/Output/Combo.cmnts.volume > TaeBlogDataProject/Combo.bodies.volume_target.bow

bin/link_lda   --train_file TaeBlogDataProject/Combo.bodies.volume_target.bow \
               --output_prefix TaeBlogDataProject/Output/Combo.bodies.volume \
               --topics 8 \
               --num_real_targets 4 \
               --model_targets | tee TaeBlogDataProject/Output/Combo.bodies.volume.log

python/view_model.py    --report=TaeBlogDataProject/Output/Combo.bodies.volume.report  \
                        --entity_names=words \
                        --dict_prefix=TaeBlogDataProject/Combo.bodies.dict. \
                        --output_prefix=TaeBlogDataProject/HTML/Combo_bodies_volume \
                        --index \
                        --topdocs \
                        --topics \
                        --docs_file TaeBlogDataProject/Combo.bodies.paintfile \
                        --docs_mode=paintlist \
                        --limit 100 \
                        --title "Combo bodies two target volumes" \
                        --target_names "RS senti,RS vol,DK senti,DK vol" \
                        --per_page 10
