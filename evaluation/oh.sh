#!/bin/bash

# calculate bandwidth overhead
#
# all parameters:
# overhead.py --in <simulated_trace> --out <overhead_result>

:<<!
# evaluate the overhead of the august machine
for pkl in 2022.10.23-15:30:47_august-5000.pkl  2022.10.23-15:42:58_august-5000.pkl  2022.10.23-15:55:11_august-5000.pkl  2022.10.23-16:07:26_august-5000.pkl  2022.10.23-16:19:36_august-5000.pkl 
do
    echo "---------------     RUN     ---------------"
    ./overhead.py --in $pkl  --out overhead-august
done
!

:<<!
# evaluate the overhead of the spring machine
for pkl in 2022.10.21-16:43:28_spring-5000.pkl  2022.10.21-17:12:21_spring-5000.pkl  2022.10.21-17:53:19_spring-5000.pkl  2022.10.21-18:13:51_spring-5000.pkl  2022.10.22-08:48:27_spring-5000.pkl 
do
    echo "---------------     RUN     ---------------"
    ./overhead.py --in $pkl  --out overhead-spring
done
!

:<<!
# evaluate the overhead of the interspace machine
for pkl in 2022.10.22-13:19:21_interspace-5000.pkl  2022.10.22-13:26:20_interspace-5000.pkl  2022.10.22-13:33:15_interspace-5000.pkl  2022.10.22-13:40:07_interspace-5000.pkl  2022.10.22-13:46:58_interspace-5000.pkl
do
    echo "---------------     RUN     ---------------"
    ./overhead.py --in $pkl  --out overhead-inter
done
!


