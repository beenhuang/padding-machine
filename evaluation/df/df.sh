#!/bin/bash

### Run DF model
#
# 1. TRAINING CASE: train/save trained DF and save test results
# df.py --train -i <ds-*.pkl> -o <res-*.csv> -sm <df-*.pkl> \
#                   
# 2. TESTING CASE: load trained DF, test DF and save result
# df.py -i <ds-*.pkl> -lm <df-*.pkl> -o <res-*.csv> \ 
#           

data=rimmer.pkl
outfile=1106-df


echo "---------------     RUN       ---------------"
./run_df.py --in $data  --out $outfile



