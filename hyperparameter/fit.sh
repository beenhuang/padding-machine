#!/bin/bash

# fit burst to distribution and save plot
#          
# burst.py --in <> --out <>
#

echo "---------------     BURST     ---------------"
./burst.py --in 2022.10.23-09:55:05_original-5000.pkl --out burst-origin

echo "---------------     INTER-ARRIVAL  TIME     ---------------"
./iat.py --in 2022.10.23-09:55:05_original-5000.pkl --out iat-origin


