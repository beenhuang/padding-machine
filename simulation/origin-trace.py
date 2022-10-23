#!/usr/bin/env python3

"""
<file>    origintrace.py
<brief>   generate original dataset&labels without running simulation
"""

import argparse
import os
import sys       
import time
from os.path import join, basename, abspath, dirname, pardir, isdir, splitext
import numpy as np
import pickle
import logging
import multiprocessing as mp

# CONSTANTS
MODULE_NAME = basename(__file__)
CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S_", time.localtime())

BASE_DIR = abspath(join(dirname(__file__), pardir))
INPUT_DIR = join(BASE_DIR, "data")
OUTPUT_DIR = join(BASE_DIR, "simulation", "sim-traces")
       
CIRCPAD_EVENT_NONPADDING_SENT = "circpad_cell_event_nonpadding_sent"
CIRCPAD_EVENT_NONPADDING_RECV = "circpad_cell_event_nonpadding_received"
CIRCPAD_ADDRESS_EVENT = "connection_ap_handshake_send_begin" 

NONPADDING_SENT = 1.0
NONPADDING_RECV = -1.0

MAX_LENGTH = 5000

def get_logger():
    logging.basicConfig(format="[%(asctime)s] >> %(message)s", level=logging.INFO)
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger

# parse arugment
def parse_arguments():
    # argument parser
    parser = argparse.ArgumentParser(description="parse arguments")

    # 1. INPUT: load ds-*.pkl dataset
    parser.add_argument("-i", "--in", required=True, metavar="<original_trace>", help="load original traces.")
    # 2. OUTPUT: save overhead in the overhead-*.txt file
    parser.add_argument("-o", "--out", required=True, metavar="<simulated_trace>", help="save simulated traces.")

    args = vars(parser.parse_args())


    return args

#
def get_files(dir):
    # c/r_file: {"ID": "file", ...}, labels: {"ID": "labels", ...}
    c_files, labels = [], {}

    # checkout 
    if not isdir(dir):
        sys.exit(f"[error]: {dir} is not a directory")
    
    # get directories
    c_mon_dir, c_unm_dir = join(dir, "client-traces", "monitored"), join(dir, "client-traces", "unmonitored")
    
    # 1. monitored traces
    for fname in os.listdir(c_mon_dir):
        ID = f"m-{splitext(fname)[0]}"

        c_files.append([ID, join(c_mon_dir, fname)])

        # file-name: site*10+page-instance
        site_page = fname.split("-")[0]

        if str(site_page)[:-1] == "" :
            labels[ID] = 0
        else:
            labels[ID] = int(str(site_page)[:-1])
   
    # 2. unmonitored traces
    max_mon_labels = max(list(labels.values()))

    for fname in os.listdir(c_unm_dir):
        ID = f"u-{splitext(fname)[0]}"

        c_files.append([ID, join(c_unm_dir, fname)])
        labels[ID] = max_mon_labels + 1 


    return c_files, labels


# extract trace
def extract_trace(ID, file, strip=True):
    with open(file, "r") as f:
        lines = f.readlines()

    # strip
    if strip:
        for idx in range(len(lines)):
            if CIRCPAD_ADDRESS_EVENT in lines[idx]:
                lines = lines[idx:]
                break

    trace = []
    
    # travel
    for idx in range(len(lines)):
        # split line:
        timestamp = int(lines[idx].split(" ")[0])
 
        # sent nonpadding case
        if CIRCPAD_EVENT_NONPADDING_SENT in lines[idx]:
            trace.append([timestamp, NONPADDING_SENT])
        # recv nonpadding case
        elif CIRCPAD_EVENT_NONPADDING_RECV in lines[idx]:
            trace.append([timestamp, NONPADDING_RECV])
        # other case    
        else:
            continue

    return (ID, np.array(trace, dtype=np.float32))

# 
def get_all_trace(c_files):
    with mp.Pool(mp.cpu_count()) as pool:
        dataset = pool.starmap(extract_trace, c_files)

    return dict(dataset)

# [MAIN]
def main():
    logger = get_logger()
    
    logger.info(f"{MODULE_NAME}: start to run.")
    #
    args = parse_arguments()
    logger.info(f"args: {args}")

    # 1. get client trace files
    c_files, labels = get_files(join(INPUT_DIR, args["in"]))
    logger.info(f"[GOT] client files.")

    # 2. extract client traces from files
    dataset = get_all_trace(c_files)
    logger.info(f"[EXTARCTED] client traces.")

    X, y = [], []
    for ID,_ in dataset.items():
        X.append(dataset[ID][:MAX_LENGTH])
        y.append(labels[ID])
        
    # 3. save original dataset&labels  
    output_file = join(OUTPUT_DIR, CURRENT_TIME+args["out"]+"-"+str(MAX_LENGTH)+".pkl")
    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)
        logger.info(f"[SAVED] original dataset,labels to the {args['out']+'.pkl'} file") 


    logger.info(f"{MODULE_NAME}: completed successfully.\n")


if __name__ == "__main__":
    sys.exit(main())
