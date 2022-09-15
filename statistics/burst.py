#!/usr/bin/env python3

"""
burststats.py

"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# parse arguments: 
parser = argparse.ArgumentParser()

# 1. INPUT: 
parser.add_argument("--in", required=True, help="load dataset from pickle file")

# 2. OUTPUT: 
parser.add_argument("--out", required=True, help="save traces to the file.")

args = vars(parser.parse_args())


# get cell count list
def get_cell_event_count(trace):

    # count: start from 1
    count = 1

    # burst list
    bursts = []

    for idx,ele in enumerate(trace[:-1]):

        # if equal, count += 1     
        if(trace[idx] == trace[idx+1]) :
            count += 1
            continue
        # if not equal, append to the bursts    
        else :
            if trace[idx] == 1.0 :
                bursts.append(count)
            elif trace[idx] == -1.0 :
                bursts.append(-count)

            # count: start from 1
            count = 1

    # the last one
    if trace[idx] == 1.0 :
        bursts.append(count)
    elif trace[idx] == -1.0 :
        bursts.append(-count)
        
    
    return bursts


def get_all_bursts(dataset):

     # write lines to the output file
    allbursts=[]

    #for ID in dataset:
    for ID, trace in dataset.items():     
        # get bursts of one trace
        bursts = get_cell_event_count(trace[0])

        # add bursts
        allbursts.extend(bursts)
        #break
    
    # compute unique & count
    burst, count = np.unique(allbursts, return_counts=True)

    return burst, count, allbursts



def main():

    print(f"----------  [{os.path.basename(__file__)}]: start to run, input: [{args['in']}]  ----------")

    # 1. load the pickle file:
    with open(os.path.join(os.getcwd(), "sim-traces", args["in"]), "rb") as f:
        dataset, _ = pickle.load(f)

    for k in dataset:
        dataset[k][0][dataset[k][0] > 1.0] = 1.0
        dataset[k][0][dataset[k][0] < -1.0] = -1.0       

    print(f"[LOADED] {args['in']} file")


    # 2. compute trace
    burst, count, allbursts = get_all_bursts(dataset)

    print(f"[GOT] all bursts, length: {len(burst)}")



    # 3. dump unique, count, allbursts:
    with open(os.path.join(os.getcwd(), "stats", args["out"]+".pkl"), "wb") as f:
        pickle.dump((burst, count, allbursts), f)

        print(f"[SAVED] unique&count&allbursts to the {args['out']}.pkl file")  


    # 4. write to the file 
    stat = dict(zip(burst, count))

    with open(os.path.join(os.getcwd(), "stats", args["out"]+".txt"), "w") as file:
        # travel
        for key, value in stat.items(): 
            file.write(f"[{key}]: [{format(value, ',')}]\n")

        #file.write(f"[ALL BURSTS] : {allbursts}")    

        print(f"[SAVED] unique&count&allbursts to the {args['out']}.txt file")   


    print(f"----------  [{os.path.basename(__file__)}]: complete successfully  ----------\n")    


if __name__ == "__main__":
    main()

