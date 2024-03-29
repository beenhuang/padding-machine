#!usr/bin/env python3

"""
<file>    exfeature.py
<brief>   extract the CUMUL feature from the trace data
"""

import numpy as np
from os.path import abspath, dirname, join

PACKET_SIZE = 514
MAX_LENGTH = 100

# 
def get_general_trace(trace):
    #
    # input: [[timestamp, direction], ...]
    #
    # output: [514, -514, ...]
    #
    gen_total = []
    direction_trace = trace[:,1].tolist()
    
    for direction in direction_trace:
        gen_total.append(int(direction)*PACKET_SIZE)

    return gen_total


# get general IN/OUT trace
def transform_general_inout_trace(trace):
    #
    gen_in = [packet for packet in trace if packet > 0]
    gen_out = [packet for packet in trace if packet < 0]

    return gen_in, gen_out 

# max_length = 100
def extract_features(trace, maxlength=MAX_LENGTH):
    if maxlength % 2 != 0 :
        sys.exit(f"[ERROR] feature_size is invalid.")

    all_features = []

    gen_total = get_general_trace(trace)
    gen_in, gen_out =  transform_general_inout_trace(gen_total)

    abs_total, cumul_total = [], []

    # travel trace
    for packet in gen_total:
        if len(cumul_total) == 0: # first element
            cumul_total.append(packet)
            abs_total.append(packet)
        else:
            cumul_total.append(cumul_total[-1] + packet)
            abs_total.append(abs_total[-1] + abs(packet))

    # cumulative feature
    cumul_feature = np.interp(np.linspace(abs_total[0], abs_total[-1], maxlength+1), abs_total, cumul_total) 


    # [2] num of in/out trace
    all_features.append(len(gen_in)) 
    all_features.append(len(gen_out)) 
    # [2] packet size of in/out trace
    all_features.append(abs(np.sum(gen_out)))
    all_features.append(abs(np.sum(gen_in)))
    # [100] cumulative feature 
    all_features.extend(cumul_feature[1:])


    return all_features    


if __name__ == "__main__":
    BASE_DIR = abspath(dirname(__file__))
    INPUT_DIR = join(BASE_DIR, "data", "output-tcp")
    FILE_NAME = "www.bild.de___-___1460455081990"

    with open(join(INPUT_DIR, FILE_NAME), "r") as f:
        lines = f.readline().rstrip("\n").split(" ")

    features = extract_features(lines[3:], 100)

    for elem in features:
        print(elem)
    

