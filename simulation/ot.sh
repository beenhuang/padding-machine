#!/bin/bash

# extract original traces
#
# all parameters:
# origin-trace.py --in <data_folder> --out <file_name> 
#

for i in {1..5}
do
    ./origin-trace.py --in standard --out original
    sleep 5m
done