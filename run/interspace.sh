#!/bin/bash

# input parameters for running simulation
simtrace=0111-interspace
machine=interspace

# evaluation output
outdir=0111
outfile=0111-interspace

for i in {1..10}
do
  # 1. run simulation
  echo "---------------     run simulation $i times    ---------------"
  ./simulation/run_simulation.py --in standard --out $simtrace-$i --machine $machine

  # 2. bandwidth overhead
  echo "---------------     bandwidth overhead $i times  ---------------"
  ./evaluation/overhead.py --in $simtrace-$i-5000.pkl  --out $outdir/overhead-$outfile

  # 3.1 cumul evaluation
  echo "---------------     CUMUL evaluation $i times   ---------------"
  ./evaluation/cumul/run_cumul.py --in $simtrace-$i-5000.pkl  --out $outdir/cumul-$outfile

  # 3.2 k-FP evaluation
  echo "---------------     k-FP evaluation $i times   ---------------"
  ./evaluation/k-FP/run_kFP.py --in $simtrace-$i-5000.pkl  --out $outdir/kfp-$outfile

  # 3.3 df evaluation
  echo "---------------     df evaluation $i times  ---------------"
  ./evaluation/df/run_df.py --in $simtrace-$i-5000.pkl  --out $outdir/df-$outfile

  echo "----------  all done $i times  ----------"
done

