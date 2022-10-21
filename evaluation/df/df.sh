#!/bin/bash

### Run DF model
#
# 1. TRAINING CASE: train/save trained DF and save test results
# df.py --train -i <ds-*.pkl> -o <res-*.csv> -sm <df-*.pkl> \
#                   
# 2. TESTING CASE: load trained DF, test DF and save result
# df.py -i <ds-*.pkl> -lm <df-*.pkl> -o <res-*.csv> \ 
#           

# TESTING:
#./df-train.py --in ds-*.pkl --out res-*.csv --model df-*.pkl 

# TRAINING:
./run_model.py --train --in 2022-10-15-19-29-46-original-trace.pkl  --out result-original-trace.csv --model original-df.pkl 
