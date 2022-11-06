#!/bin/bash

out=1104-august-a

:<<!
# 1. run simulation
for i in {1..5}
do
    echo "---------------     run simulation in $i times     ---------------"
    ./simulation/run_simulation.py --in standard --out $out-$i --machine august
done


# bandwidth overhead
for i in {1..5}
do
    echo "---------------     bandwidth overhead in $i times   ---------------"
    ./evaluation/overhead.py --in $out-$i-5000.pkl  --out overhead-$out-$i
done


# 2. evaluation
# 2.1 cumul
for i in {1..5}
do
    echo "---------------     CUMUL evaluation in $i times    ---------------"
    ./evaluation/cumul/run_cumul.py --in $out-$i-5000.pkl  --out cumul-$out-$i
done

# 2.1 k-FP
for i in {1..5}
do
    echo "---------------     k-FP evaluation in $i times    ---------------"
    ./evaluation/k-FP/run_kFP.py --in $out-$i-5000.pkl  --out kfp-$out-$i
done
!

# 2.3 df
for i in {1..5}
do
    echo "---------------     df evaluation in $i times    ---------------"
    ./evaluation/df/run_df.py --in $out-$i-5000.pkl  --out df-$out-$i
done

echo "----------   done   ----------"

