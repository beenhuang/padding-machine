#!/bin/bash

# run simulation, produced dataset&label and save to the pickle file
#
# simulate's all arguments:
# simulate.py -i <input_dir> -o <simulated_traces> -m <padding_machine>
#


./simulate.py -i standard -o spring -m spring
./simulate.py -i standard -o interspace -m interspace
./simulate.py -i standard -o august -m august