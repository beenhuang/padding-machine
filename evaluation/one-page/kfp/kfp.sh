#!/bin/bash

# all parameters of k-FP model :
#./run_kFP.py --in <input_dir> --out <metrics>

#infile=0111-october-8-5000.pkl

outdir=0125
#outfile=onepage-kfp-october-8

#echo "---------------     one-page setting evaluation    ---------------"
#./run_kFP.py --in $infile  --out $outdir/$outfile

for i in {1..2}
do
  echo "---------------    one-page setting evaluation for the $i times  ---------------"
  ./run_kFP.py --in 0111-original-$i-5000.pkl  --out $outdir/onepage-kfp-original-$i

  echo "----------  end for $i time  ----------"
done
