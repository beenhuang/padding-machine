#!/usr/bin/env python3

"""

  origintrace.py

  generate original_dataset(ds-origin.pkl) from circuit_padding_trace to cell_trace 
  used for the df attack

"""

import argparse
import os 
import sys       
import pickle

import numpy as np


# argument parser:
parser = argparse.ArgumentParser()

# 1. input: client padding event trace folder
parser.add_argument("--in", required=True, help="extract dataset, path with {monitored,unmonitored} subfolders")

# 2. output: the pickle file named ds-origin.pkl
parser.add_argument("--out", required=True, help="save dataset&label")

# 3. dataset arguments: class:partition:sample
parser.add_argument("--class", required=False, type=int, 
                    default=50, help="the number of monitored classes")
parser.add_argument("--part", required=False, type=int, 
                    default=10, help="the number of monitored partitions")
parser.add_argument("--sample", required=False, type=int, 
                    default=20, help="the number of monitored samples")    
# 4. max length:                  
parser.add_argument("--length", required=False, type=int, 
                    default=5000, help="max input length")

args = vars(parser.parse_args())

# macros:         
CIRCPAD_EVENT_NONPADDING_SENT = "circpad_cell_event_nonpadding_sent"
CIRCPAD_EVENT_NONPADDING_RECV = "circpad_cell_event_nonpadding_received"
CIRCPAD_EVENT_PADDING_SENT = "circpad_cell_event_padding_sent"
CIRCPAD_EVENT_PADDING_RECV = "circpad_cell_event_padding_received"

CIRCPAD_ADDRESS_EVENT = "connection_ap_handshake_send_begin"         

'''

  get trace directory:
  -- parameter: dir
  -- return value: dir/monitored, dir/unmonitored  

'''    
def get_trace_directory(dir):
    
    # checkout if dir is a directory?
    if not os.path.isdir(dir):
        sys.exit(f"[error]: {dir} is not a directory")
        
    # monintored dictorary.
    mon_dir = os.path.join(dir, "monitored")
    
    # checkout if mon_dir is a directory?
    if not os.path.isdir(mon_dir):
        sys.exit(f"[error]: {mon_dir} is not a directory")

    # unmonitored dictorary.
    unm_dir = os.path.join(dir, "unmonitored")

    # checkout if unm_dir is a directory?
    if not os.path.isdir(unm_dir):
        sys.exit(f"[error]: {unm_dir} is not a directory")

    return mon_dir, unm_dir    

'''
trace2cell.py:

  convert simulation event trace to a cell direction sequence

'''
def trace2cell(log, length, strip=False):

    # initialize nparray for cell sequence
    cells = np.zeros((1, length), dtype=np.float32)
    iats = np.zeros((1, length), dtype=np.int64)

    # split log
    lines = log.split("\n")

    # strip
    if strip:
        for i, line in enumerate(lines):
            if CIRCPAD_ADDRESS_EVENT in line:
                lines = lines[i:]
                break


    # number of cell
    n = 0
    
    # travel
    for line in lines:
        # split line:
        element = line.split(" ")
 
        # 1. nonpadding sent = 1.0
        if CIRCPAD_EVENT_NONPADDING_SENT in line:
            cells[0][n] = 1.0
            iats[0][n] = element[0]
            n += 1
        # 2. nonpadding received = -1.0
        elif CIRCPAD_EVENT_NONPADDING_RECV in line:
            cells[0][n] = -1.0
            iats[0][n] = element[0]
            n += 1
        else:
            continue

        # max_lengthgth
        if n == length:
            break                      
        
    return cells, iats

'''
  load_dataset.py:

  load dataset/label from monitored/unmonitored directory

'''
def load_dataset(mon_dir, unm_dir, classes, parts, samples, length, extract_func):

    dataset = {}
    iats = {}
    label = {}

    # 1. monitored dataset.
    # traverse classes.
    for cls in range(0, classes):
        # traverse partitions.
        for part in range(0, parts):
            # traverse samples.
            for smpl in range(0, samples):
                # 1. ID: monitored dataset ID
                ID = f"m-{cls}-{part}-{smpl}"
                
                # 2. label[ID}: ID:class
                label[ID] = cls

                # 3. dataset[ID]: 
                with open(os.path.join(mon_dir, f"{cls*10+part}-{smpl}.trace"), "r") as f:
                    dataset[ID], iats[ID] = extract_func(f.read(), length)

                    # [TEST] one trace
                    #return dataset, iats, label

    # 2. load unmonitored dataset, equal to the number of monitored dataset
    flist = os.listdir(unm_dir)[:len(dataset)]

    for fname in flist:
        # 1. ID: unmonitored dataset ID
        ID = f"u-{os.path.splitext(fname)[0]}"

        # 2. label[ID]: load label, start from 0 for monitored
        label[ID] = classes 

        # 3. dataset[ID]: load unmonitored dataset:
        with open(os.path.join(unm_dir, fname), "r") as f:
            dataset[ID], iats[ID] = extract_func(f.read(), length)

    return dataset, iats, label

def main():

    print(f"-------  [{os.path.basename(__file__)}]: start to run [{args['in']}]  -------")

    # 1. get client monitored/unmonitored directory.
    mon_dir, unm_dir = get_trace_directory(os.path.join(os.getcwd(), "dataset", "standard", args["in"]))
    
    print(f"[GOT] client monitored/unmonitored directory")

    
    # 2. load dataset & iats & labels
    dataset, iats, label = load_dataset(mon_dir, unm_dir, 
                                        args["class"], args["part"], args["sample"],
                                        args["length"], trace2cell)

    print(f"[EXTARCTED] dataset,iats,labels")


    # 3. dump original dataset,iates,label
    with open(os.path.join(os.getcwd(), "sim-traces", args["out"]+".pkl"), "wb") as f:
        pickle.dump((dataset, iats, label), f)

        print(f"[SAVED] original dataset,iats,label to the {args['out']+'.pkl'} file") 


    # 4. write original dataset,iates,label
    lines=[]

    for ID in dataset:
        lines.append(f"[{ID}]: ")
        lines.append(",".join(f'[{iats[ID][0][i]},{a}]' for i, a in enumerate(dataset[ID][0])))
        lines.append(f";\n\n")

    with open(os.path.join(os.getcwd(), "sim-traces", args["out"]+".txt"), "w") as f:
        f.writelines(lines)  
        
        print(f"[SAVED] original dataset,iats,label to the {args['out']+'.txt'} file")       


    print(f"-------  [{os.path.basename(__file__)}]: completed successfully  -------\n\n")


if __name__ == "__main__":
    main()
