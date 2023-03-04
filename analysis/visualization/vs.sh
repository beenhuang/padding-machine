#!/bin/bash


infile=1114-august-15-15-15-3-5000.pkl
outfile=august-15-15-15-3

# visualization
echo "---------------     visualize    ---------------"
./visualize.py --in $infile  --out $outfile
