#!/bin/bash

### Run DF model
#
# 1. TRAINING CASE: train/save trained DF and save test results
# df.py --train -i <ds-*.pkl> -o <res-*.csv> -sm <df-*.pkl> \
#                   
# 2. TESTING CASE: load trained DF, test DF and save result
# df.py -i <ds-*.pkl> -lm <df-*.pkl> -o <res-*.csv> \ 
#           


for pkl in 2022.10.23-15:30:47_august-5000.pkl  2022.10.23-15:42:58_august-5000.pkl 2022.10.23-15:55:11_august-5000.pkl 2022.10.23-16:07:26_august-5000.pkl 2022.10.23-16:19:36_august-5000.pkl
do
    echo "---------------     RUN       ---------------"
    ./run_df.py --in $pkl  --out df-august
done

:<<!
# TRAINING:
for pkl in 2022.10.22-13:19:21_interspace-5000.pkl          2022.10.22-13:26:20_interspace-5000.pkl    2022.10.22-13:33:15_interspace-5000.pkl    2022.10.22-13:40:07_interspace-5000.pkl   2022.10.22-13:46:58_interspace-5000.pkl  
do
    echo "---------------     RUN     ---------------"
    ./run_df.py --in $pkl  --out df-interspace
done
!
