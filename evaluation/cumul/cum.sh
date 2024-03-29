#!/bin/bash

# all parameters of CUMUL model :
#./run_cumul.py --in <input_dir> --out <metrics>


:<<!
# evaluate the august machine using k-FP classifier
for pkl in 2022.10.23-15:30:47_august-5000.pkl  2022.10.23-15:42:58_august-5000.pkl 2022.10.23-15:55:11_august-5000.pkl 2022.10.23-16:07:26_august-5000.pkl 2022.10.23-16:19:36_august-5000.pkl
do
    echo "---------------     RUN  AUGUST     ---------------"
    ./run_cumul.py --in $pkl  --out cumul-august
done
!


# evaluate the interspace machine using k-FP classifier
for pkl in 2022.10.22-13:19:21_interspace-5000.pkl          2022.10.22-13:26:20_interspace-5000.pkl    2022.10.22-13:33:15_interspace-5000.pkl    2022.10.22-13:40:07_interspace-5000.pkl   2022.10.22-13:46:58_interspace-5000.pkl  
do
    echo "---------------     RUN  INTERSPACE     ---------------"
    ./run_cumul.py --in $pkl  --out cumul-interspace
    break
done


:<<!
for pkl in 2022.10.21-16:43:28_spring-5000.pkl 2022.10.21-17:12:21_spring-5000.pkl 2022.10.21-17:53:19_spring-5000.pkl 2022.10.21-18:13:51_spring-5000.pkl 2022.10.22-08:48:27_spring-5000.pkl
do
    echo "---------------     RUN  SPRING     ---------------"
    ./run_cumul.py --in $pkl  --out cumul-spring
done
!

:<<!
# evaluate original traces using k-FP classifier without any padding machines.
for pkl in 2022.10.23-09:55:05_original-5000.pkl 2022.10.23-09:57:21_original-5000.pkl 2022.10.23-09:59:38_original-5000.pkl 2022.10.23-10:01:55_original-5000.pkl 2022.10.23-10:04:12_original-5000.pkl  
do
    echo "---------------     RUN  ORIGINAL     ---------------"
    ./run_cumul.py --in $pkl  --out cumul-original
done
!
