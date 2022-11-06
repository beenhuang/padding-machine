#!/bin/bash

# run simulation, produced dataset&label and save to the pickle file
#
# simulate's all arguments:
# simulate.py -i <input_dir> -o <simulated_traces> -m <padding_machine>

:<<!
# run the spring machine
for i in {1..5}
do
    echo "---------------     run $i times     ---------------"
    ./run_simulation.py --in standard --out spring --machine spring
    sleep 5m
done
!

:<<!
# run the interspace machine
for i in {1..5}
do
    echo "---------------     run $i times     ---------------"
    ./run_simulation.py --in standard --out interspace --machine interspace
    sleep 5m
done
!

# run the august machine
for i in {1..5}
do
    echo "---------------     run $i times     ---------------"
    ./run_simulation.py --in standard --out august --machine august
    #sleep 5m
done

