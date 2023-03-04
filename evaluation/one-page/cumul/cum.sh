#!/bin/bash

# all parameters of CUMUL model:
#./run_cumul.py --in <input_file> --out <result>

infile=0111-rbb-2-5000.pkl
outfile=one-page-cumul-october

echo "---------------     one-page setting evaluation    ---------------"
./run_cumul.py --in $infile  --out $outfile
