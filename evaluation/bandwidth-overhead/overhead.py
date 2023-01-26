#!/usr/bin/env python3

"""
<file>    overhead.py
<brief>   evaluate bandwidth overhead for the defended trace.
"""

import argparse
import os
import sys
import time
import pickle
import logging
import numpy as np
from os.path import join, basename, abspath, splitext, dirname, pardir, isdir


# CONSTANTS
MODULE_NAME = basename(__file__)
CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S_", time.localtime())

BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
INPUT_DIR = join(BASE_DIR, "simulation", "sim-traces")
OUTPUT_DIR = join(BASE_DIR, "results")

NONPADDING_SENT = 1.0
NONPADDING_RECV = -1.0
PADDING_SENT = 2.0
PADDING_RECV = -2.0


def get_logger():
    logging.basicConfig(format="[%(asctime)s] >> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger

def parse_arguments():
    # argument parser
    parser = argparse.ArgumentParser(description="k-FP")

    # 1. INPUT: (pickle file) load the dataset defended by the padding machine.
    parser.add_argument("-i", "--in", required=True, metavar="<simulated-traces>", help="load simulated traces from a pickle file")
    # 2. OUTPUT: (text file) save bandwidth overhead in the text file.
    parser.add_argument("-o", "--out", required=True, metavar="<result-file>", help="save bandwidth overhead to the text file")

    args = vars(parser.parse_args())

    return args


#
def get_cell_count(dataset):
    # sent/recv nonpadding cells:
    sent_nonpadding, recv_nonpadding = 0, 0
    # sent/recv padding cells:
    sent_padding, recv_padding = 0, 0

    for trace in dataset:
        # compute unique & count
        unique, count = np.unique(trace[:, 1], return_counts=True)

        # create unique:count {1:num, -1:num, 2:num, -2:num}
        cell_count = dict(zip(unique, count))

        # sent_nonpadding == 0:
        if cell_count[NONPADDING_SENT] == 0:
            sys.exit(f"[WARN] sent nonpadding cells: 0")
        # recv_nonpadding == 0
        elif cell_count[NONPADDING_RECV] == 0:  
            sys.exit(f"[WARN] recv nonpadding cells: 0")

        sent_nonpadding += cell_count[NONPADDING_SENT]
        recv_nonpadding += cell_count[NONPADDING_RECV]

        if PADDING_SENT in cell_count:
            sent_padding += cell_count[PADDING_SENT]

        if PADDING_RECV in cell_count:
            recv_padding += cell_count[PADDING_RECV]    


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

    lines.append(f"total trace: {format(len(dataset), ',')} \n\n")
    
    # all element 
    lines.append(f"sent_nonpadding: {format(sent_nonpadding, ',')} , sent_padding: {format(sent_padding, ',')} \n")
    lines.append(f"recv_nonpadding: {format(recv_nonpadding, ',')} , recv_padding: {format(recv_padding, ',')} \n\n")
    
    # sent/recv/total
    lines.append(f"sent cells: {format(sent, ',')} , sent/(sent+recv)= {float(sent)/float(sent+recv):.0%} \n")
    lines.append(f"recv cells: {format(recv, ',')} , recv/(sent+recv)= {float(recv)/float(sent+recv):.0%} \n")
    lines.append(f"total cells: {format(sent+recv, ',')} \n\n")

    # sent/recv bandwidth average
    lines.append(f"Avg sent bandwidth: sent/sent_nonpadding= {avg_sent:.0%} \n")
    lines.append(f"Avg recv bandwidth: recv/recv_nonpadding= {avg_recv:.0%} \n\n")

    # total bandwidth average
    lines.append(f"Avg total bandwidth: (sent+recv)/(sent_nonpadding+recv_nonpadding)= {avg_total:.0%}\n\n\n") 


    return lines   


# main function
def main(input, output, logger):
    """
    Run the bandwidth overhead evaluation on a defended trace.

    Parameters
    ----------
    input: str
        Operating system file path to the directory containing processed feature files.
    output: str
        Operating system file path to the directory where analysis results should be saved.
    logger : object
        logger object is used to log the message when running the program.
    """

    # [1] load the defended traces named <input>.
    input_file = join(INPUT_DIR, input)
    with open(input_file, "rb") as f:
        dataset, _ = pickle.load(f)
    logger.info(f"[LOADED] dataset from the {args['in']}")    


    # [2] calculate bandwidth overhead
    sent_nonpadding, sent_padding, recv_nonpadding, recv_padding = get_cell_count(dataset)
    lines = get_bandwidth_overhead(dataset, sent_nonpadding, sent_padding, recv_nonpadding, recv_padding)
    logger.info(f"[CALCULATED] bandwidth overhead")
    

    # [3] write bandwidth overhead to the file named <output>.
    output_file = join(OUTPUT_DIR, output+".txt")
    with open(output_file, "a") as f:
        f.writelines(lines)
        logger.info(f"[SAVED] bandwidth overhead in the {output}")


    logger.info(f"{MODULE_NAME}: complete successfully.\n")
        

if __name__ == "__main__":
    try:
        # generate logger
        logger = get_logger()
        logger.info(f"{MODULE_NAME}: start to run.")

        # parse arguments
        args = parse_arguments()
        logger.info(f"Arguments: {args}")
        
        main(args["in"], args["out"], logger)
    except KeyboardInterrupt:
        sys.exit(-1)
