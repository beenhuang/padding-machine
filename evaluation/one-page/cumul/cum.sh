#!/bin/bash

# all parameters of CUMUL model:
#./run_cumul.py --in <input_file> --out <result>

infile=2022.10.23-15:30:47_august-5000.pkl
outfile=cumul-august

echo "---------------     CUMUL evaluation    ---------------"
./evaluation/cumul/run_cumul.py --in $infile  --out $outfile
