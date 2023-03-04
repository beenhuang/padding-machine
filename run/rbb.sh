#!/bin/bash

# input parameters for running simulation
simtrace=0111-rbb
machine=rbb

# evaluation output
outdir=0111
outfile=0111-rbb

for i in {1..10}
do
  # 1. run simulation
  echo "---------------    start running simulation for the $i time    ---------------"
  ./simulation/run_simulation.py --in standard --out $simtrace-$i --machine $machine

  # 2. bandwidth overhead
  echo "---------------     bandwidth overhead for the $i time  ---------------"
  ./evaluation/overhead.py --in $simtrace-$i-5000.pkl  --out $outdir/overhead-$outfile

  # 3.1 cumul evaluation
  echo "---------------     CUMUL for the $i time   ---------------"
  ./evaluation/cumul/run_cumul.py --in $simtrace-$i-5000.pkl  --out $outdir/cumul-$outfile

  # 3.2 k-FP evaluation
  echo "---------------     k-FP for the $i time   ---------------"
  ./evaluation/k-FP/run_kFP.py --in $simtrace-$i-5000.pkl  --out $outdir/kfp-$outfile

  # 3.3 df evaluation
  echo "---------------     DF for the  $i time  ---------------"
  ./evaluation/df/run_df.py --in $simtrace-$i-5000.pkl  --out $outdir/df-$outfile

  echo "----------   $i time end ----------"
  
done

