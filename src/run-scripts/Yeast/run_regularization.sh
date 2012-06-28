i=Yeast
bin/link_lda  --link_train_file $i/edges.blah \
              --topics 15 \
              --iterations 50 \
              --output_prefix $i/OutputModels/baseline

bin/link_lda  --link_train_file $i/edges.blah \
              --topics 15 \
              --output_prefix $i/OutputModels/volume \
              --iterations 50 \
              --volume_penalty 4 \
              --volume_variance 0.5

bin/link_lda  --link_train_file $i/edges.blah \
              --topics 15 \
              --output_prefix $i/OutputModels/role \
              --iterations 50 \
              --mixed_penalty 1 \
              --mixed_variance 0.5

bin/link_lda  --link_train_file $i/edges.blah \
              --topics 15 \
              --output_prefix $i/OutputModels/both \
              --iterations 50 \
              --mixed_penalty 1 \
              --mixed_variance 0.5 \
              --volume_penalty 4 \
              --volume_variance 0.5
