#!/usr/bin/env python3

''' 
<file>    template.py
<brief>   brief of thie file
'''

import argparse
import configparser
import time
import sys
import os
import logging
import subprocess
import multiprocessing as mp
import numpy as np
import pickle
from os.path import join, isdir, splitext, basename, abspath, dirname, pardir, isfile

# CONSTANTS
MODULE_NAME = basename(__file__)
CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S_", time.localtime())

BASE_DIR = abspath(join(dirname(__file__), pardir))
INPUT_DIR = join(BASE_DIR, "data")
OUTPUT_DIR = join(BASE_DIR, "simulation", "sim-traces")
CONFIG_DIR = join(BASE_DIR, "simulation")
MACHINE_DIR = join(BASE_DIR, "machines")
TOR_DIR = abspath(join(BASE_DIR, pardir))

# client/relay token in the test_circuitpadding_sim.c
CLIENT_MACHINE_TOKEN = "//REPLACE-client-padding-machine-REPLACE"
RELAY_MACHINE_TOKEN = "//REPLACE-relay-padding-machine-REPLACE"

# 
NONPADDING_SENT = 1.0
NONPADDING_RECV = -1.0
PADDING_SENT = 2.0
PADDING_RECV = -2.0

CIRCPAD_EVENT_NONPADDING_SENT = "circpad_cell_event_nonpadding_sent"
CIRCPAD_EVENT_NONPADDING_RECV = "circpad_cell_event_nonpadding_received"
CIRCPAD_EVENT_PADDING_SENT = "circpad_cell_event_padding_sent"
CIRCPAD_EVENT_PADDING_RECV = "circpad_cell_event_padding_received"
CIRCPAD_ADDRESS_EVENT = "connection_ap_handshake_send_begin"

#
def get_logger():
    logging.basicConfig(format="%(asctime)s>> %(message)s", level=logging.INFO)
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
    # 3. Padding Machine: load specified padding machines
    parser.add_argument("-m", "--padding_machine", required=True, metavar="<padding_machine>", help="load specified padding machines.")

    args = vars(parser.parse_args())

    # configuration parser
    config_file = join(CONFIG_DIR, "config.ini")

    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)
    padding_machine = args["padding_machine"]


    return args, config_parser[padding_machine]

def add_machines(client_machine, relay_machine, tor):
    #
    test_sim_file = join(tor, "src", "test", "test_circuitpadding_sim.c")

    if not isfile(test_sim_file):
        sys.exit(f"[ERROR]: {test_sim_file} doses not exist")

    # load the client circuit_padding_machine:
    with open(client_machine, "r") as f:
        mc = f.read()
    # load the relay circuit_padding_machine:
    with open(relay_machine, "r") as f:
        mr = f.read()
    # load original simulation module
    with open(test_sim_file, "r") as f:
        origin_sim_module = f.read()

    # assert origin_sim_module loaded from test_circuitpadding_sim.c
    assert(origin_sim_module != "")
    assert(CLIENT_MACHINE_TOKEN in origin_sim_module)
    assert(RELAY_MACHINE_TOKEN in origin_sim_module)

    # add client&relay machine
    modified_module = origin_sim_module.replace(CLIENT_MACHINE_TOKEN, mc)
    modified_module = modified_module.replace(RELAY_MACHINE_TOKEN, mr)
        
    # write modified_module contents to the test_circuitpadding_sim.c
    with open(test_sim_file, "w") as f:
        f.write(modified_module)

    # run make command
    cmd = f"cd {tor} && make"
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
    assert(result.returncode == 0) 

    # 3. resotre original_circpad_module
    with open(test_sim_file, "w") as f:
        f.write(origin_sim_module)    


def run_simualte(todo):
    with mp.Pool(mp.cpu_count()) as pool:
        result = pool.starmap(run_simulate_once, todo)
        print(f"[COMPLETED] all simulation experiments.")

    return dict(result)

def run_simulate_once(ID, c_file, r_file):
    # run the Tor unit test
    cmd = f"{join(tor, 'src/test/test circuitpadding_sim/..')} --info --circpadsim {c_file} {r_file} 247 {max_length}"
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    # verify the result
    if result.returncode != 0:
        print(f"[ERROR] return_code: {result.returncode} for [{cmd}]")

    # log list which has client/relay padding event trace
    event_log = result.stdout.split("\n")
    client_trace = extract_client_trace(event_log)

    return (ID, client_trace)

def extract_client_trace(log):
    trace = []

    for line in log:
        # skip not "source=client" 
        if not "source=client" in line:
            continue

        # split line:
        element = line.split(" ")[5]
        timestamp = int(element.split("=")[1])
        
        #print(f"line: {line}")
 
        # nonpadding sent case
        if CIRCPAD_EVENT_NONPADDING_SENT in line:
            trace.append([timestamp, NONPADDING_SENT])
        # nonpadding recv case
        elif CIRCPAD_EVENT_NONPADDING_RECV in line:
            trace.append([timestamp, NONPADDING_RECV])
        # padding sent                
        elif CIRCPAD_EVENT_PADDING_SENT in line:
            trace.append([timestamp, PADDING_SENT])
        # padding received
        elif CIRCPAD_EVENT_PADDING_RECV in line:
            trace.append([timestamp, PADDING_RECV])
        else:
            pass
            #print(f"[ERROR]: {line}")        

    return trace        




#
def get_trace_files(dir):
    # c/r_file: {"ID": "file", ...}, labels: {"ID": "labels", ...}
    files, labels = [], {}

    # checkout 
    if not isdir(dir):
        sys.exit(f"[error]: {dir} is not a directory")
    
    # get directories
    c_mon_dir, c_unm_dir = join(dir, "client-traces", "monitored"), join(dir, "client-traces", "unmonitored")
    r_mon_dir, r_unm_dir = join(dir, "fakerelay-traces", "monitored"), join(dir, "fakerelay-traces", "unmonitored")

    # 1. monitored traces
    # [f for f in os.listdir(c_mon_dir) if not f.startswith('.')]
    for fname in os.listdir(c_mon_dir):
        #
        ID = f"m-{splitext(fname)[0]}"

        files.append([ID, join(c_mon_dir, fname), join(r_mon_dir, fname)])

        # file-name: site*10+page-instance
        site = fname.split("-")[0]
        if str(site)[:-1] == "" :
            labels[ID] = 0
        else:
            labels[ID] = int(str(site)[:-1])
    
    # 2. unmonitored traces
    max_mon_labels = max(list(labels.values()))

    for fname in os.listdir(c_unm_dir)[:len(files)]:
        # 
        ID = f"u-{splitext(fname)[0]}"

        files.append([ID, join(c_unm_dir, fname), join(r_unm_dir, fname)])
        
        labels[ID] = max_mon_labels + 1 
    

    return files, labels


def main():
    logger = get_logger()
    logger.info(f"{MODULE_NAME}: start to run.")

    # parse arguments
    args, config = parse_arguments()
    logger.info(f"args: {args}, config: {config}")

    # 1. get client/relay trace-files and labelss
    files, labels = get_trace_files(join(INPUT_DIR, args["in"]))
    logger.info(f"[LOADED] {len(labels)} client/fake_relay files.")

    # 2. add client/relay machine to the test_circuitpadding_sim.c file
    client_machine = join(MACHINE_DIR, config["client_machine"])
    relay_machine = join(MACHINE_DIR, config["relay_machine"])
    global tor 
    tor = join(TOR_DIR, config["tor"])
    
    add_machines(client_machine, relay_machine, tor)
    logger.info(f"[ADDED] {config['client_machine']} & {config['relay_machine']   } machines.")


    # 3. run the circpad-sim simulator.
    logger.info("[RUNNING] simulation")
    global max_length
    max_length = config["max_length"]
    # run the simulation
    dataset = run_simualte(files)


    # 4. save trace
    X, y = [], []

    for ID,trace in dataset.items():
        X.append(np.array(trace, dtype=np.float32))
        y.append(labels[ID])

    output_file = join(OUTPUT_DIR, CURRENT_TIME+args["out"]+"-"+max_length+".pkl")
    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)
        logger.info(f"[SAVED] original dataset,labels to the {args['out']+'.pkl'} file") 
        

    logger.info(f"{MODULE_NAME}: completed successfully.\n")  


if __name__ == "__main__":
    sys.exit(main())