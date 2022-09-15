#
# run DF model
#
#
# 1. TRAINING CASE: train DF, save trained DF, test DF and save result
# 
# df.py --train --ld <ds-*.pkl> --csv <res-*.csv> \
#       --sm <df-*.pkl> \
#       --epoch <num:30> --batchsize <num:750> \
#       --class <num:50> --part <num:10> --sample <num:20> \
#       --fold <num:0-9> 
#                   
#
# 2. TESTING CASE: load trained DF, test DF and save result
# 
# df.py --ld <ds-*.pkl> --csv <res-*.csv> \ 
#       --lm <df-*.pkl> \
#       --batchsize <num:750> \
#                  

# TRAINING CASE: train the new DF model
# evaluation/df.py --train --ld ds-*.pkl --sm df-*.pkl --csv res-*.csv

# TESTING CASE: test the trained DF model
# evaluation/df.py --ld ds-*.pkl --lm df-*.pkl --csv res-*.csv

evaluation/df.py --ld ds-inter.pkl --lm df-inter.pkl --csv res-inter.csv
