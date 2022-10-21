#!/bin/bash

# calculate bandwidth overhead
#
# all parameters:
# overhead.py --in <simulated_trace> --out <overhead_result>
#

# machine spring:
./overhead.py --in 2022-10-17-11:28:43-spring.pkl  --out overhead-spring

# machine interspace:
./overhead.py --in 2022-10-17-10:11:54-interspace.pkl  --out overhead-inter

# machine august:
./overhead.py --in  2022-10-17-15:19:24-august.pkl  --out overhead-august
