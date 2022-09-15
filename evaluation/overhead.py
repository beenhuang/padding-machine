#!/usr/bin/env python3

"""overhead.py

compute bandwidth overhead.

Bandwidth overhead is based on number of padding and non-padding cells in all traces.

"""

import argparse
import os
import sys
import pickle
import numpy as np

# parse arguments: 
parser = argparse.ArgumentParser()

# 1. INPUT: load ds-*.pkl dataset
parser.add_argument("--in", required=True, help="load dataset from pickle file")

# 2. OUTPUT: save overhead in the overhead-*.txt file
parser.add_argument("--out", required=True, help="save overhead to the file.")

args = vars(parser.parse_args())


#
def get_cell_count(dataset):

    # sent/recv nonpadding cells:
    sent_nonpadding = 0
    recv_nonpadding = 0

    # sent/recv padding cells:
    sent_padding = 0
    recv_padding = 0

    # 
    for trace in dataset:

        # compute unique & count
        unique, count = np.unique(dataset[trace][0], return_counts=True)

        # create unique:count {1:num, -1:num, 2:num, -2:num}
        cell_count = dict(zip(unique, count))

        # sent_nonpadding == 0:
        if cell_count[1] == 0:
            sys.exit(f"[WARN] sent nonpadding cells: [0]")
        # recv_nonpadding == 0
        elif cell_count[-1] == 0:  
            sys.exit(f"[WARN] recv nonpadding cells: [0]")


        sent_nonpadding += cell_count[1]
        recv_nonpadding += cell_count[-1]

        if 2 in cell_count:
            sent_padding += cell_count[2]

        if -2 in cell_count:
            recv_padding += cell_count[-2]    


    return sent_nonpadding, sent_padding, recv_nonpadding, recv_padding  


#
def get_bandwidth_overhead(dataset, sent_nonpadding, sent_padding, recv_nonpadding, recv_padding):

    # total sent cells        
    sent = sent_padding + sent_nonpadding

    # total recev cells
    recv = recv_padding + recv_nonpadding

    # average sent/recv
    avg_sent = float(sent)/float(sent_nonpadding)
    avg_recv = float(recv)/float(recv_nonpadding)

    # average total
    avg_total = float(sent+recv)/float(sent_nonpadding+recv_nonpadding)

    # lines write to the over-*.txt file
    lines = []

    lines.append(f"total trace: [{format(len(dataset), ',')}] \n\n")
    
    # all element 
    lines.append(f"sent_nonpadding: [{format(sent_nonpadding, ',')}] , sent_padding: [{format(sent_padding, ',')}] \n")
    lines.append(f"recv_nonpadding: [{format(recv_nonpadding, ',')}] , recv_padding: [{format(recv_padding, ',')}] \n\n")
    
    # sent/recv/total
    lines.append(f"sent cells: [{format(sent, ',')}] , sent/(sent+recv)= [{float(sent)/float(sent+recv):.0%}] \n")
    lines.append(f"recv cells: [{format(recv, ',')}] , recv/(sent+recv)= [{float(recv)/float(sent+recv):.0%}] \n")
    lines.append(f"total cells: [{format(sent+recv, ',')}] \n\n")

    # sent/recv bandwidth average
    lines.append(f"Avg sent bandwidth: sent/sent_nonpadding= [{avg_sent:.0%}] \n")
    lines.append(f"Avg recv bandwidth: recv/recv_nonpadding= [{avg_recv:.0%}] \n\n")

    # total bandwidth average
    lines.append(f"Avg total bandwidth: (sent+recv)/(sent_nonpadding+recv_nonpadding)= [{avg_total:.0%}]") 


    return lines   


# main function:
def main():

    print(f"-------  [{os.path.basename(__file__)}]: start to run, input: [{args['in']}]  -------")

    # [1] load dataset & label:
    with open(os.path.join(os.getcwd(), "sim-traces", args["in"]), "rb") as f:
        dataset, _ = pickle.load(f)

    print(f"[LOADED] dataset, label from the [{args['in']}]")    


    # [2] calculate bandwidth overhead
    sent_nonpadding, sent_padding, recv_nonpadding, recv_padding = get_cell_count(dataset)
    
    lines = get_bandwidth_overhead(dataset, sent_nonpadding, sent_padding, recv_nonpadding, recv_padding)

    print(f"[CALCULATED] bandwidth overhead")
    

    # [3] write bandwidth overhead to the file.
    with open(os.path.join(os.getcwd(), "results", args["out"]), "w") as f:
        f.writelines(lines)
        
        print(f"[SAVED] bandwidth overhead in the {args['out']}")
        
        
    print(f"-------  [{os.path.basename(__file__)}]: complete successfully  -------")

if __name__ == "__main__":
    main()