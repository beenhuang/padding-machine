#!/usr/bin/env python3

"""
<file>    burst.py
<brief>   
"""

import argparse
import os
import sys   
import time
import pickle
import itertools
import logging
import numpy as np
import multiprocessing as mp
from os.path import join, basename, abspath, dirname, pardir, isdir, splitext 

from dist_fit import fit_data_to_dist
from plot import save_bar_plot, show_bar_plot2


# CONSTANTS
MODULE_NAME = basename(__file__)
CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

BASE_DIR = abspath(join(dirname(__file__), pardir))
INPUT_DIR = join(BASE_DIR, "simulation", "sim-traces")
OUTPUT_DIR = join(BASE_DIR, "hyperparameter", "fit-results", "burst")

NONPADDING_SENT = 1.0
NONPADDING_RECV = -1.0

PADDING_SENT = 2.0
PADDING_RECV = -2.0


#
def get_logger():
    logging.basicConfig(format="[%(asctime)s]>>> %(message)s", level=logging.INFO, datefmt = "%Y-%m-%d %H:%M:%S")
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


def get_burst_trace(trace):
    trace = trace[:,1]
    #print(f"trace: {trace}")
    trace[trace == PADDING_SENT] = NONPADDING_SENT
    trace[trace == PADDING_RECV] = NONPADDING_RECV   
    #print(f"new_trace: {trace}")

    return [sum(list(group)) for _,group in itertools.groupby(trace)]

    #return [sum(list(group)) for _,group in itertools.groupby(trace[:,1])]


def transform_to_burst_trace(dataset):
    with mp.Pool(mp.cpu_count()) as pool:
        result = pool.map(get_burst_trace, dataset)

    return result    


def mon_unmon_split(bursts, labels):
    unmon_index = labels.index(max(labels))

    mon_bursts = bursts[:unmon_index]
    unmon_bursts = bursts[unmon_index:]

    return mon_bursts, unmon_bursts                 

#
def make_burst_plot(bursts, file):
    all_bursts = np.concatenate(bursts).astype(int)
    send_burst = all_bursts[all_bursts > 0]
    recv_burst = -all_bursts[all_bursts < 0] 

    unique_total, count_total = np.unique(all_bursts, return_counts=True)
    unique_send, count_send = np.unique(send_burst, return_counts=True)
    unique_recv, count_recv = np.unique(recv_burst, return_counts=True)

    with open(join(OUTPUT_DIR, f"{CURRENT_TIME}_{file}.txt"), "w") as f:
        for line in list(zip(unique_total, count_total)):
            f.write(str(line)+"\n") 

    show_bar_plot2(unique_send[:15], count_send[:15], "SEND BURST", "burst size", "count")
    show_bar_plot2(unique_recv[:15], count_recv[:15], "RECV BURST", "burst size", "count")     

    # save
    for max_index in range(10, 210, 10):
        save_bar_plot(unique_send[:max_index], count_send[:max_index], join(OUTPUT_DIR, f"{CURRENT_TIME}_send-{max_index}-{file}.png"), "SEND BURST", "send burst", "count")
        save_bar_plot(unique_recv[:max_index], count_recv[:max_index], join(OUTPUT_DIR, f"{CURRENT_TIME}_recv-{max_index}-{file}.png"), "RECV BURST", "recv burst", "count")     


#
def get_burst_distfit(bursts):
    all_bursts = np.concatenate(bursts).astype(int)
    send_burst = all_bursts[all_bursts > 0]
    recv_burst = -all_bursts[all_bursts < 0] 
    
    recv_max = 6
    #recv_burst = recv_burst[recv_burst > 9]
    recv_burst = recv_burst[recv_burst < recv_max]
    
    send_max = 6
    send_burst = send_burst[send_burst < send_max]
    
    todo = [[send_burst, "uniform"], [send_burst, "logistic"],
            [send_burst, "fisk"], [send_burst, "geom"],
            [send_burst, "weibull"],[send_burst, "pareto"],
            [recv_burst, "uniform"], [recv_burst, "logistic"],
            [recv_burst, "fisk"],[recv_burst, "geom"],
            [recv_burst, "weibull"],[recv_burst, "pareto"]]

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(fit_data_to_dist, todo)
        
    results.insert(0, "\n-----  send_burst  -----\n")
    results.insert(7, "\n-----  recv_burst  -----\n")

    return results
    
# [MAIN]
def main():
    logger = get_logger()
    logger.info(f"{MODULE_NAME}: start to run.")
    
    args = parse_arguments()
    logger.info(f"Arguments: {args}")

    # 1. load the dataset:
    with open(join(INPUT_DIR, args["in"]), "rb") as f:
        dataset, labels = pickle.load(f)
    logger.info(f"LOADED dataset&labels, length:{len(dataset)}")

    # 2. get burst trace {"ID":[burst_list], ...}
    bursts = transform_to_burst_trace(dataset)
    mon_bursts, unmon_bursts = mon_unmon_split(bursts, labels)
    logger.info(f"TRANSFORMED burst_trace, total_length:{len(bursts)}, mon_length:{len(mon_bursts)}, unmon_length:{len(unmon_bursts)}")
    

    # 3. save burst plot
    logger.info(f"MAKING burst plots ...")
    make_burst_plot(bursts, f"all-{args['out']}")
    #make_burst_plot(mon_bursts, f"mon-{args['out']}")
    #make_burst_plot(unmon_bursts, f"unmon-{args['out']}")    
    logger.info(f"MAKED burst plots.")
    sys.exit(0)

    # 4. fit data to distribution
    logger.info(f"FITTING burst to the distribution ...")
    dist_total = get_burst_distfit(bursts)
    dist_mon = get_burst_distfit(mon_bursts)
    dist_unmon = get_burst_distfit(unmon_bursts)

    with open(join(OUTPUT_DIR, f"{CURRENT_TIME}_distfit-{args['out']}.txt"), "w") as f:
        f.write(f"<<<<<<<<<< all bursts >>>>>>>>>>\n")
        f.writelines(dist_total)
        f.write(f"\n\n<<<<<<<<<< monitored bursts >>>>>>>>>>\n")
        f.writelines(dist_mon)
        f.write(f"\n\n<<<<<<<<<< unmonitored bursts >>>>>>>>>>\n")
        f.writelines(dist_unmon)
        logger.info(f"[SAVED] distribution fit results")   
        

    logger.info(f"{MODULE_NAME}: complete successfully.\n")    


if __name__ == "__main__":
    sys.exit(main())


