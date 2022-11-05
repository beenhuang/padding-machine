#!/bin/bash

# extract original traces
#
# all parameters:
# origin-trace.py --in <data_folder> --out <file_name> 

for i in {1..5}
do
    echo "---------------     run $i times     ---------------"
    ./origin-trace.py --in standard --out original  --maxlength 5000
    break
    sleep 5m
done