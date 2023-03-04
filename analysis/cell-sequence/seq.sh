#!/bin/bash

# write cell sequence
#
# all parameters:
# sequence.py --in <defended_trace> --out <output_directory>

input=0228-march-0110-0110-5000.pkl
output=march-07

echo "---------------     RUN     ---------------"

./sequence.py --in $input  --out $output

echo "---------------     DONE     ---------------"
