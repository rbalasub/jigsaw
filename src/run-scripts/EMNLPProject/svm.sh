for i in books dvd electronics kitchen movies
do
  N=`cat EMNLPProject/$i.dict.words | wc -l`
  for V in Y2 #Y N     #Y - WordReg, Y2=ThetaReg  (from jake and elwood resp.)
  do
    python python/add_lda_feats_svm.py $N < EMNLPProject/$i.svm.${V}reg.topic_features | paste -d' ' EMNLPProject/$i.svm "-" | 
      python python/rand_split.py > EMNLPProject/$i.$V.svm.train 2> EMNLPProject/$i.$V.svm.test

    ~/ResearchTools/svm_light/svm_learn -z r  -t 0 EMNLPProject/$i.$V.svm.train EMNLPProject/Output/$i.$V.svm.model
    ~/ResearchTools/svm_light/svm_classify EMNLPProject/$i.$V.svm.test EMNLPProject/Output/$i.$V.svm.model EMNLPProject/Output/$i.$V.svm.pred
    echo -n $i ' ' >> EMNLPProject/svmres.$V 
    cut -f1 -d' ' EMNLPProject/$i.$V.svm.test | paste "-"  EMNLPProject/Output/$i.$V.svm.pred | python python/compute_mse.py >> EMNLPProject/svmres.$V
  done
done

