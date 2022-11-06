#!/usr/bin/env python3

"""
<file>    iat.py
<brief> 
"""

import argparse
import os
from os.path import join, basename, abspath, dirname, pardir, isdir, splitext 
import sys   
import time
import pickle
import logging
import numpy as np
import multiprocessing as mp

from dist_fit import fit_data_to_dist
from plot import save_bar_plot, show_bar_plot


# CONSTANTS
MODULE_NAME = basename(__file__)
CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

BASE_DIR = abspath(join(dirname(__file__), pardir))
INPUT_DIR = join(BASE_DIR, "simulation", "sim-traces")
OUTPUT_DIR = join(BASE_DIR, "hyperparameter", "fit-results", "iat")

DIRECTION_OUT = 1.0
DIERCTION_IN = -1.0

# 
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

# s2s, s2r, r2s, r2r iat trace
def get_iat_trace(trace):
    # sent-to-sent/sent-to-recv/recv-to-send/recv-to-recv iat trace
    s2s, s2r, r2s, r2r = [], [], [], []

    # timestamp: trace[idx][0], direction: trace[idx][1]
    for idx in range(len(trace)-1): 
        # 1. send-to-send case
        if (trace[idx][1] == DIRECTION_OUT and trace[idx+1][1] == DIRECTION_OUT) :
            # convert nanosecond to microsecond
            iat = int((trace[idx+1][0]-trace[idx][0]) * 0.001)
            s2s.append(iat)
        # 2. send-to-recv case
        elif (trace[idx][1] == DIRECTION_OUT and trace[idx+1][1] == DIERCTION_IN) :
            # convert nanosecond to microsecond
            iat = int((trace[idx+1][0]-trace[idx][0]) * 0.001)
            s2r.append(iat)
        # 3. recv-to-send case
        elif (trace[idx][1] == DIERCTION_IN and trace[idx+1][1] == DIRECTION_OUT) :
            # convert nanosecond to microsecond
            iat = int((trace[idx+1][0]-trace[idx][0]) * 0.001)
            r2s.append(iat)
        # 4. recv-to-recv case
        elif (trace[idx][1] == DIERCTION_IN and trace[idx+1][1] == DIERCTION_IN) :
            # convert nanosecond to microsecond
            iat = int((trace[idx+1][0]-trace[idx][0]) * 0.001)
            r2r.append(iat)
        else:
            sys.exit(f"[EEROR] unrecognized direction value.") 
    
    return [s2s, s2r, r2s, r2r]

def transform_to_iat_trace(dataset):
    with mp.Pool(mp.cpu_count()) as pool:
        result = pool.map(get_iat_trace, dataset)
        
    return result    

def mon_unmon_split(iats, labels):
    unmon_index = labels.index(max(labels))

    mon_iats = iats[:unmon_index]
    unmon_iats = iats[unmon_index:]

    return mon_iats, unmon_iats  

#
def make_iat_plot(iats, file):
    s2s, r2r, s2r, r2s = [], [], [], []
    for elem in iats:
        s2s.extend(elem[0])
        r2r.extend(elem[3])
        #s2r.extend(elem[1])
        #r2s.extend(elem[2])

    unique_s2s, count_s2s = np.unique(np.array(s2s, dtype=np.int32), return_counts=True)
    unique_r2r, count_r2r = np.unique(np.array(r2r, dtype=np.int32), return_counts=True)


    with open(join(OUTPUT_DIR, f"{CURRENT_TIME}_{file}.txt"), "w") as f:
        f.write(f"----- send-to-send -----\n")
        for line in list(zip(unique_s2s, count_s2s)):
            f.write(str(line)+"\n")
        f.write(f"----- recv-to-recv -----\n")
        for line in list(zip(unique_r2r, count_r2r)):
            f.write(str(line)+"\n")    

    # show
    #show_bar_plot(unique_s2s[:30], count_s2s[:30], "SEND-TO-SEND IAT", "send-to-send", "count")
    #show_bar_plot(unique_r2r[:30], count_r2r[:30], "RECV-TO-RECV IAT", "recv-to-recv", "count")     

    # save
    for max_index in range(10, 210, 10):
        save_bar_plot(unique_s2s[:max_index], count_s2s[:max_index], join(OUTPUT_DIR, f"{CURRENT_TIME}_s2s-{max_index}-{file}.png"), "SEND-TO-SEND IAT", "send-to-send", "count")
        save_bar_plot(unique_r2r[:max_index], count_r2r[:max_index], join(OUTPUT_DIR, f"{CURRENT_TIME}_r2r-{max_index}-{file}.png"), "RECV-TO-RECV IAT", "recv-to-recv", "count")

#
def fit_iat_to_dist(iats):
    s2s, r2r, s2r, r2s = [], [], [], []
    for elem in iats:
        s2s.extend(elem[0])
        r2r.extend(elem[3])
        #s2r.extend(elem[1])
        #r2s.extend(elem[2])

    s2s = np.array(s2s, dtype=np.int32)
    r2r = np.array(r2r, dtype=np.int32)    

    todo = [[s2s, "uniform"], [s2s, "logistic"],
            [s2s, "fisk"], [s2s, "geom"],
            [s2s, "weibull"], [s2s, "pareto"],
            [r2r, "uniform"], [r2r, "logistic"],
            [r2r, "fisk"], [r2r, "geom"],
            [r2r, "weibull"], [r2r, "pareto"]]

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(fit_data_to_dist, todo)

    results.insert(0, "\n-----  send_to_send   -----\n")
    results.insert(7, "\n-----  recv_to_recv   -----\n")

    return results

# [MAIN]
def main():
    logger = get_logger()
    logger.info(f"{MODULE_NAME}: start to run.")
    
    args = parse_arguments()
    logger.info(f"args: {args}")

    # 1. load the dataset:
    with open(join(INPUT_DIR, args["in"]), "rb") as f:
        dataset, labels = pickle.load(f)
    logger.info(f"LOADED dataset&labels, length:{len(dataset)}")

    # 2. get iat trace
    iats = transform_to_iat_trace(dataset)
    mon_iats, unmon_iats = mon_unmon_split(iats, labels)
    logger.info(f"TRANSFORMED iat_trace, total_length:{len(iats)}, mon_length:{len(mon_iats)}, unmon_length:{len(unmon_iats)}")

    # 3. 
    logger.info(f"MAKING iat plots ...")
    make_iat_plot(iats, "all-"+args["out"]) 
    make_iat_plot(mon_iats, "mon-"+args["out"])
    make_iat_plot(unmon_iats, "unmon-"+args["out"])  
    logger.info(f"MAKED iat plots.")

    # 4. fit iat to distribution
    logger.info(f"FITTING iats to the distribution ...")
    dist_total = fit_iat_to_dist(iats)
    dist_mon = fit_iat_to_dist(mon_iats)
    dist_unmon = fit_iat_to_dist(unmon_iats)

    with open(join(OUTPUT_DIR, f"{CURRENT_TIME}_distfit-{args['out']}.txt"), "w") as f:
        f.write(f"<<<<<<<<<< all iats >>>>>>>>>>\n")
        f.writelines(dist_total)
        f.write(f"\n\n<<<<<<<<<< monitored iats >>>>>>>>>>\n")
        f.writelines(dist_mon)
        f.write(f"\n\n<<<<<<<<<< unmonitored iats >>>>>>>>>>\n")
        f.writelines(dist_unmon)
        logger.info(f"SAVED distribution fit results")   


    logger.info(f"{MODULE_NAME}: complete successfully.\n")    


if __name__ == "__main__":
    sys.exit(main())


