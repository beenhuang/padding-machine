#!/bin/bash

# calculate bandwidth overhead
#
# all parameters:
# overhead.py --in <simulated_trace> --out <overhead_result>

input=0111-october-7-5000.pkl
output=october

echo "---------------     RUN     ---------------"

./overhead.py --in $input  --out $output

echo "---------------     DONE     ---------------"
