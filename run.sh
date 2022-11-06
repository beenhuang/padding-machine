#!/bin/bash

simtrace=1106-august

machinedir=1104
machine=august

outdir=1106


# 1. run simulation
echo "---------------     run simulation     ---------------"
./simulation/run_simulation.py --in standard --out $simtrace --machine $machine

# 2. bandwidth overhead
echo "---------------     bandwidth overhead   ---------------"
./evaluation/overhead.py --in $simtrace-5000.pkl  --out $outdir/overhead-$simtrace

# 3.1 cumul evaluation
echo "---------------     CUMUL evaluation    ---------------"
./evaluation/cumul/run_cumul.py --in $simtrace-5000.pkl  --out $outdir/cumul-$simtrace

# 3.2 k-FP evaluation
echo "---------------     k-FP evaluation    ---------------"
./evaluation/k-FP/run_kFP.py --in $simtrace-5000.pkl  --out $outdir/kfp-$simtrace

# 3.3 df evaluation
echo "---------------     df evaluation   ---------------"
./evaluation/df/run_df.py --in $simtrace-5000.pkl  --out $outdir/df-$simtrace

echo "----------  all done   ----------"


