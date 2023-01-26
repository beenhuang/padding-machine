#!/usr/bin/env python3

"""
<file>    get-dataset.py
<brief>   generate original dataset&labels from Rimmer's dataset
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

NONPADDING_SENT = 1.0
NONPADDING_RECV = -1.0


def get_logger():
    logging.basicConfig(format="[%(asctime)s] >> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
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

def get_data():
    mon_file = join(INPUT_DIR, args["in"], "tor_500w_2500tr.npz")
    unmon_file = join(INPUT_DIR, args["in"], "tor_open_400000w.npz")
    
    mon = np.load(mon_file, allow_pickle=True)
    unmon = np.load(unmon_file, allow_pickle=True)

    for label in mon["labels"]:
        print(label)



# [MAIN]
def main():
    logger = get_logger()
    logger.info(f"{MODULE_NAME}: start to run.")

    args = parse_arguments()
    logger.info(f"args: {args}")

    get_data()

    #mon_file = join(INPUT_DIR, args["in"], "tor_500w_2500tr.npz")
    #unmon_file = join(INPUT_DIR, args["in"], "tor_open_400000w.npz")
    #mon = np.load(mon_file, allow_pickle=True)
    #unmon = np.load(unmon_file, allow_pickle=True)
    
    #print(mon.files)
    #print(mon["data"].shape)
    #print(mon["labels"].shape)
    #print(unmon["data"].shape)
    #print(unmon["labels"].shape)    
    #print(mon["labels"])
 
    logger.info(f"[GOT] client files.")



        
    # 3. save original dataset&labels  
    #output_file = join(OUTPUT_DIR, args["out"]+"-"+str(args["maxlength"])+".pkl")
    #output_file = join(OUTPUT_DIR, f"{args['out']}.pkl")
    #with open(output_file, "wb") as f:
    #    pickle.dump((X, y), f)
    #    logger.info(f"[SAVED] original dataset,labels to the {args['out']+'.pkl'} file") 


    logger.info(f"{MODULE_NAME}: completed successfully.\n")


if __name__ == "__main__":
    sys.exit(main())
