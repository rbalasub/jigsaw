PREFIX=
Run () {
  suffix=${PREFIX}seed-0.theta-$3-$2.mixed-$5-$4
  bin/link_lda --train_file EMNLPProject/$1.slda.bow \
               --output_prefix EMNLPProject/Output/$1.$suffix \
               --topics 20 \
               --runs 2 \
               --randinit -1 \
               --num_real_targets 1 \
               --theta_penalty $3  \
               --theta_variance $2 \
               --mixed_penalty $5  \
               --mixed_variance $4 \
               --model_targets | tee EMNLPProject/$1.$suffix.log
}

RunWithLabels () {
  suffix=${PREFIX}seed-1.theta-$3-$2.mixed-$5-$4
  bin/link_lda --train_file EMNLPProject/$1.slda.bow \
               --runs 2 \
               --randinit -1 \
               --output_prefix EMNLPProject/Output/$1.$suffix \
               --node_label_file EMNLPProject/$1.seed_labels \
               --topics 20 \
               --num_real_targets 1 \
               --label_randomness 0.05 \
               --clamp_rigidity 0.95 \
               --theta_penalty $3  \
               --theta_variance $2 \
               --mixed_penalty $5  \
               --mixed_variance $4 \
               --model_targets | tee EMNLPProject/$1.$suffix.log
}

View () {
  python/view_model.py    --report=EMNLPProject/Output/$1.report  \
                          --entity_names=words \
                          --dict_prefix=EMNLPProject/$2 \
                          --output_prefix=EMNLPProject/HTML/$1 \
                          --index  \
                          --topics \
                          --title "$1" \
                          --target_names "Stars" 
}

for role in 0.2
do
  if [[ $role == '0.0' ]]
  then
    role_flag=0
  else
    role_flag=1
  fi
  for theta in 0.0 # 0.5 0.2
  do
    if [[ $theta == '0.0' ]]
    then
      theta_flag=0
    else
      theta_flag=1
    fi
    for seeds in 0 1 
    do
      for b in movies books dvd electronics kitchen
      do
        if [[ $seeds == 0 ]]
        then
           echo "What"
          Run         $b   $theta  $theta_flag   $role $role_flag
        else
          RunWithLabels   $b $theta  $theta_flag   $role $role_flag 
        fi
        suffix=${PREFIX}seed-$seeds.theta-$theta_flag-$theta.mixed-$role_flag-$role
        View        $b.$suffix $b.dict.
        echo $suffix >> EMNLPProject/combined.res; grep avg.train_target_0_mse EMNLPProject/Output/$b.$suffix.report >> EMNLPProject/combined.res
      done #dataset
    done #seeds
  done #role
done #theta


