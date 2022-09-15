#!/usr/bin/env python3

''' run the simulator with client/relay traces and 
    client/relay circuit padding machines.

  This script runs the following steps:

  1. get client/relay all traces files.

  2. add client/relay padding machines in the test_circuitpadding_sim.c

  3. < main > : run the circpad-sim simulation.

  4. save the dataset using pickle module.

'''

import argparse
import sys
import os
import subprocess
import signal
import pickle
from multiprocessing import Pool

import numpy as np


# argument parser
parser = argparse.ArgumentParser()

# INPUT : client and relay original circuitpadding event traces.
parser.add_argument("--tc", required=True, help="input folder of client trace files")
parser.add_argument("--tr", required=True, help="input folder of relay trace files")

# INPUT : client-side and relay-side circuit padding machines.
parser.add_argument("--mc", required=True, help="path to file of client machine")
parser.add_argument("--mr", required=True, help="path to file of relay machine")

# INPUT : tor directory path
parser.add_argument("--tor", required=True, help="path to tor folder (~/tor-0.4.7.8)")

# OUTPUT : save client simulated dataset sim.pkl
parser.add_argument("--out", required=True, help="file to save results to")

# OUTPUT_parameter : the max length of cells.
parser.add_argument("--length", required=False, type=int, 
                    default=5000, help="max length of extracted cells")

# DATASET_parameter : class(website) * partition(webpage) * sample = total dataset.
parser.add_argument("--class", required=False, type=int, 
                    default=50, help="the number of monitored classes")
parser.add_argument("--part", required=False, type=int, 
                    default=10, help="the number of parts")
parser.add_argument("--sample", required=False, type=int, 
                    default=20, help="the number of samples")

# OPTION : run in parallel
parser.add_argument("--worker", required=False, type=int, 
                    default=8, help="number of workers for simulating machines")


args = vars(parser.parse_args())

# constants
CIRCPAD_EVENT_NONPADDING_SENT = "circpad_cell_event_nonpadding_sent"
CIRCPAD_EVENT_NONPADDING_RECV = "circpad_cell_event_nonpadding_received"
CIRCPAD_EVENT_PADDING_SENT = "circpad_cell_event_padding_sent"
CIRCPAD_EVENT_PADDING_RECV = "circpad_cell_event_padding_received"
CIRCPAD_ADDRESS_EVENT = "connection_ap_handshake_send_begin"

# client/relay token in the test_circuitpadding_sim.c
CLIENT_MACHINE_TOKEN = "//REPLACE-client-padding-machine-REPLACE"
RELAY_MACHINE_TOKEN = "//REPLACE-relay-padding-machine-REPLACE"

# tor unit test command
TOR_CIRCPADSIM_CMD = os.path.join(os.pardir, args["tor"], "src/test/test circuitpadding_sim/..")
TOR_CIRCPADSIM_CMD_FORMAT = f"{os.path.join(os.pardir, args['tor'], 'src/test/test circuitpadding_sim/..')} --info --circpadsim {{}} {{}} 247 {{}}"

# the original contents of the test_circuitpadding_sim.c
origin_sim_module = "" 

# ~/tor-0.4.7.8 + src/test/test_circuitpadding_sim.c
test_sim_file = os.path.join(os.pardir, args["tor"], "src", "test", "test_circuitpadding_sim.c")


#
def get_trace_directory(dir):
    
    # checkout if dir is a directory?
    if not os.path.isdir(dir):
        sys.exit(f"[error]: {dir} is not a directory")
        
    # 1. monintored dictorary.
    mon_dir = os.path.join(dir, "monitored")
    
    # checkout if mon_dir is a directory?
    if not os.path.isdir(mon_dir):
        sys.exit(f"[error]: {mon_dir} is not a directory")

    # 2. unmonitored dictorary.
    unm_dir = os.path.join(dir, "unmonitored")

    # checkout if unm_dir is a directory?
    if not os.path.isdir(unm_dir):
        sys.exit(f"[error]: {unm_dir} is not a directory")

    return mon_dir, unm_dir     


#
def get_trace_files(c_dir, r_dir, classes, parts, samples):

    # label: {"ID": "label", ...}
    label = {}
    # client-file: {"ID": "file", ...}
    c_file = {}
    # relay-file: {"ID": "file", ...}
    r_file = {}
    
    # get c_mon_dir/c_unm_dir/r_mon_dir/r_unm_dir directories
    c_mon_dir, c_unm_dir = get_trace_directory(c_dir)
    r_mon_dir, r_unm_dir = get_trace_directory(r_dir)

    # 1. monitored traces.
    # traverse classes.
    for cls in range(0, classes):
        # traverse parts.
        for part in range(0, parts):
            # traverse samples.
            for smpl in range(0, samples):
                # ID
                ID = f"m-{cls}-{part}-{smpl}"
                
                # 1. label[ID}: ID:class
                label[ID] = cls

                # 2. c_file[ID] & r_file[ID]: ID:file_name
                fname = f"{cls*10+part}-{smpl}.trace"

                c_file[ID] = os.path.join(c_mon_dir, fname)
                r_file[ID] = os.path.join(r_mon_dir, fname)

                # verify if the client file exists?
                if not os.path.exists(c_file[ID]):
                    sys.exit(f"[error] : {c_file[ID]} does not exist")

                # verify if the relay file exists?
                if not os.path.exists(r_file[ID]):
                    sys.exit(f"[error] : {r_file[ID]} does not exist")

    # 2. unmonitored traces.
    flist = os.listdir(c_unm_dir)[:len(label)]

    for fname in flist:
        # ID
        ID = f"u-{os.path.splitext(fname)[0]}"

        # 1. label[ID]: ID:class
        label[ID] = classes 

        # 2. c_file[ID] & r_file[ID] : ID:file_name
        c_file[ID] = os.path.join(c_unm_dir, fname)
        r_file[ID] = os.path.join(r_unm_dir, fname)

        # verify if the client file exists?
        if not os.path.exists(c_file[ID]):
            sys.exit(f"[error] : {c_file[ID]} does not exist")
            
        # verify if the relay file exists?
        if not os.path.exists(r_file[ID]):
            sys.exit(f"[error] : {r_file[ID]} does not exist")
    
    return label, c_file, r_file  

def add_machines(mc_file, mr_file, tor):
    
    # load client/relay machine
    # load the client circuit_padding_machine:
    with open(mc_file, "r") as f:
        mc = f.read()

    # load the relay circuit_padding_machine:
    with open(mr_file, "r") as f:
        mr = f.read()
    
    # load test_circuitpadding_sim.c contents to origin_sim_module
    global origin_sim_module, test_sim_file

    if origin_sim_module == "":
        with open(test_sim_file, "r") as f:
            origin_sim_module = f.read()

    # assert origin_sim_module loaded from test_circuitpadding_sim.c
    assert(origin_sim_module != "")
    assert(CLIENT_MACHINE_TOKEN in origin_sim_module)
    assert(RELAY_MACHINE_TOKEN in origin_sim_module)

    # 1. replace with machines and save the modified file.
    # add client machine
    modified_module = origin_sim_module.replace(CLIENT_MACHINE_TOKEN, mc)
    # add relay machine
    modified_module = modified_module.replace(RELAY_MACHINE_TOKEN, mr)

    # write modified_module contents to the test_circuitpadding_sim.c
    with open(test_sim_file, "w") as f:
        f.write(modified_module)

    # 2. run make command
    cmd = f"cd {tor} && make"
    
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)

    assert(result.returncode == 0) 

    # 3. resotre original_circpad_module
    
    # write the module to file.
    with open(test_sim_file, "w") as f:
        f.write(origin_sim_module)


def extract_cellsequ(log, need_c_cells=True):

    i = 0

    # max length: default is 50,000
    length = args["length"]

    # return value: cell sequence
    cell_sequ = np.zeros((1, length), dtype=np.float32)
    iats = np.zeros((1, length), dtype=np.int64)

    for line in log:
        if i >= length:
            break
        
        # skip not "source=client" 
        if need_c_cells and not "source=client" in line:
            continue

        # skip not "source=relay"
        elif not need_c_cells and not "source=relay" in line:
            continue

        # split line:
        element = line.split(" ")    

        # nonpadding sent
        if CIRCPAD_EVENT_NONPADDING_SENT in line:
            cell_sequ[0][i] = 1.0 
            iats[0][i] = element[0]
        # nonpadding received    
        elif CIRCPAD_EVENT_NONPADDING_RECV in line:
            cell_sequ[0][i] = -1.0
            iats[0][i] = element[0]
        # padding sent                
        elif CIRCPAD_EVENT_PADDING_SENT in line:
            cell_sequ[0][i] = 2.0
            iats[0][i] = element[0]
        # padding received
        elif CIRCPAD_EVENT_PADDING_RECV in line:
            cell_sequ[0][i] = -2.0
            iats[0][i] = element[0]

        # increate i 
        i += 1

    return cell_sequ, iats


'''

1. run the simulation once.
2. extract client/relay cell sequence from result log list.

'''
def run_simulate_once(one_c_file, one_r_file, ID, extract_func, length, need_c_dataset=True, need_r_dataset=False):

    # generate simulation command.
    cmd = TOR_CIRCPADSIM_CMD_FORMAT.format(one_c_file, one_r_file, length)

    # 1. run the Tor unit test
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)

    # verify the result
    if result.returncode != 0:
        print(f"run_simulate_once(): return_code: [{result.returncode}] for [{cmd}]")

    # extract client/relay cell sequence from the simulated result
    c_cellsequ = []
    r_cellsequ = []

    # log list which has client/relay padding event trace
    event_log = result.stdout.split("\n")

    # 2. extract client cell sequence from combined log list
    if need_c_dataset:
        c_cellsequ, c_iats = extract_func(event_log, need_c_cells=True)

    # 2. extract relay cell sequence from combined log list
    if need_r_dataset:
        r_cellsequ, r_iats = extract_func(event_log, need_c_cells=False)

    return (ID, c_cellsequ, c_iats, r_cellsequ, r_iats)         

'''

  run the simulator

'''

def run_simulate_machines(label, c_files, r_files, extract_func, worker, length, need_c_dataset=True, need_r_dataset=False):

    # to do list
    todo = []

    for ID in label:
        todo.append((c_files[ID], r_files[ID], ID, extract_func, length, need_c_dataset, need_r_dataset))

    # process pool
    pool = Pool(worker)
    
    # return value: ID, c_cellsequ, r_cellsequ
    result = pool.starmap(run_simulate_once, todo)

    print(f"[COMPLETED] all simulation experiments.")
    
    # ID:cell_sequence(1.0, 2.0, -1.0, -2.0)
    c_dataset = {}
    c_iats = {}
    r_dataset = {}
    r_iats = {}

    # ID, c_cellsequ, c_iats, r_cellsequ, r_iats
    for res in result:
        # client dataset: ID=cellsequ
        if need_c_dataset:
            c_dataset[res[0]] = res[1]
            c_iats[res[0]] = res[2]
        # relay dataset: ID=cellsequ    
        if need_r_dataset:
            r_dataset[res[0]] = res[3]
            r_iats[res[0]] = res[4]
    
    print(f"[EXTRACTED] client/relay simluated dataset")
    
    pool.close()

    return c_dataset, c_iats, r_dataset, r_iats


def main():
    
    print(f"-------  [{os.path.basename(__file__)}]: start  -------")

    # properly restore tor source when closed
    signal.signal(signal.SIGINT, sigint_handler)

    # 1. get label/client/relay trace files
    label, c_files, r_files = get_trace_files(os.path.join(os.getcwd(), "dataset", "standard", args["tc"]), os.path.join(os.getcwd(), "dataset", "standard", args["tr"]), args["class"], args["part"], args["sample"])

    print(f"[LOADED] {len(label)} client/fake_relay files.")


    # 2. add client/relay machine to the test_circuitpadding_sim.c file
    add_machines(os.path.join(os.getcwd(), "machines", args["mc"]), os.path.join(os.getcwd(), "machines", args["mr"]), os.path.join(os.pardir, args["tor"]))

    print(f"[ADDED] {args['mc']} & {args['mr']} machines.")


    # 3. run the circpad-sim simulator.
    print("[RUNNING] simulation")

    # run the simulation
    c_dataset, c_iats, _, _ = run_simulate_machines(label, c_files, r_files, extract_cellsequ, args["worker"], args["length"])


    # 4. save the client dataset to the trace-*.pkl file
    with open(os.path.join(os.getcwd(), "sim-traces", args["out"]), "wb") as f: 
        pickle.dump((c_dataset, c_iats, label), f)

        print(f"[SAVED] client simulated dataset to [{args['out']}] file")


    print(f"-------  [{os.path.basename(__file__)}]: completed successfully  -------")


if __name__ == "__main__":
    main()
