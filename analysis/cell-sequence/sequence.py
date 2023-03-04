#!/usr/bin/env python3

"""
<file>    sequence.py
<brief>   write cell sequence.
"""

import argparse
import os
import sys
import time
import pickle
import logging
import numpy as np
from os.path import join, basename, abspath, splitext, dirname, pardir, isdir, exists


# CONSTANTS
MODULE_NAME = basename(__file__)
CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S_", time.localtime())

BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
INPUT_DIR = join(BASE_DIR, "simulation", "sim-traces")
OUTPUT_DIR = join(BASE_DIR, "analysis", "cell-sequence", "result")

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
def cell_sequence(dataset, labels, out_dir):
    #data = zip(dataset, labels)
    n = 0 

    for idx,trace in enumerate(dataset):
        n += 1
        logger.debug(f"trace: {trace}")
        new_trace = []
        for packet in trace:
            logger.debug(f"packet: {packet}")
            if packet[1] == NONPADDING_SENT:
                new_trace.append("+")
            elif packet[1] == NONPADDING_RECV:
                new_trace.append("-")
            elif packet[1] == PADDING_SENT:
                new_trace.append("P")
            elif packet[1] == PADDING_RECV:
                new_trace.append("V")
            else:
                logger.info(f"unrecognized packets: {packet}")

        
        logger.debug(f"new_trace: {new_trace}")

        output_file = join(out_dir, str(labels[idx]))
        logger.debug(f"output_file: {output_file}")
        with open(output_file, "a") as f:
            f.writelines(new_trace)
            f.write("\n\n")
        
        if n == 10:
            sys.exit(0)    
        
    logger.info(f"[COMPLETE]")

# main function
def main(input, output):
    """
    write cell sequences.

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
    
    with open(input, "rb") as f:
        dataset, labels = pickle.load(f)
    logger.debug(f"[LOADED] dataset dataset:{dataset}, labels:{labels}")    

    # [2] calculate bandwidth overhead
    cell_sequence(dataset, labels, output)

    logger.info(f"{MODULE_NAME}: complete successfully.\n")
        

if __name__ == "__main__":
    try:
        # generate logger
        logger = get_logger()
        logger.info(f"{MODULE_NAME}: start to run.")

        # parse arguments
        args = parse_arguments()
        logger.info(f"Arguments: {args}")

        # input data file.
        in_file = join(INPUT_DIR, args["in"])

        # make "result" directory
        if not exists(OUTPUT_DIR):
          os.makedirs(OUTPUT_DIR)

        # create output directory if not exists.
        out_dir = join(OUTPUT_DIR, args["out"])
        if not exists(out_dir):
          os.makedirs(out_dir)

        main(in_file, out_dir)
    except KeyboardInterrupt:
        sys.exit(-1)
