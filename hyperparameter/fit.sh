#!/bin/bash

# <brief>: fit burst to distribution and save plot      
# ./burst.py --in <input> --out <output>


burst_in=0111-original-1-5000.pkl
burst_out=original-burst

echo "---------------     BURST     ---------------"
./burst.py --in $burst_in --out $burst_out


iat_in=2022.10.23-09:55:05_original-5000.pkl
iat_out=iat-origin-10-30

#echo "---------------     INTER-ARRIVAL  TIME     ---------------"
#./iat.py --in $iat_in --out $iat_out


