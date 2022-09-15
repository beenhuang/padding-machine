#!/usr/bin/env python3

"""
iats.py

"""

import argparse
import os
import pickle
import numpy as np


# parse arguments: 
parser = argparse.ArgumentParser()

# 1. INPUT: 
parser.add_argument("--in", required=True, help="load dataset from pickle file")

# 2. OUTPUT: 
parser.add_argument("--out", required=True, help="save traces to the file.")

args = vars(parser.parse_args())


# get iats for one trace
def get_iats_for_one_trace(trace, iats):

    # sent-to-sent iat list
    ss = []
    # sent-to-recv iat list
    sr = []
    # recv-to-sent iat list
    rs = []
    # recv-to-recv iat list
    rr = []

    # travel
    # for i in range(1, len(trace)):

    for idx,ele in enumerate(trace[:-1]):
        # trace
        if (trace[idx] == 1 and trace[idx+1] == 1) :
            # convert nanosecond to microsecond
            iat = int(str(iats[idx+1]-iats[idx])[:-3])
            ss.append(iat)

        elif (trace[idx] == 1 and trace[idx+1] == -1) :
            # convert nanosecond to microsecond
            iat = int(str(iats[idx+1]-iats[idx])[:-3])
            sr.append(iat)

        elif (trace[idx] == -1 and trace[idx+1] == 1) :
            # convert nanosecond to microsecond
            iat = int(str(iats[idx+1]-iats[idx])[:-3])
            rs.append(iat)

        elif (trace[idx] == -1 and trace[idx+1] == -1) :
            # convert nanosecond to microsecond
            iat = int(str(iats[idx+1]-iats[idx])[:-3])
            rr.append(iat)
        else:
            #print(f"index: {idx}, element: {ele}, {trace[idx+1]}")
            break    

    return ss, sr, rs, rr    

# get iats for one trace
def get_all_iats(dataset, iats):  

    # iats for all traces
    all_ss=[]
    all_sr=[]
    all_rs=[]
    all_rr=[]

    #trace
    for ID, trace in dataset.items():     
        # get iats of one trace
        ss, sr, rs, rr = get_iats_for_one_trace(trace[0], iats[ID][0])

        # add iats
        all_ss.extend(ss)
        all_sr.extend(sr)
        all_rs.extend(rs)
        all_rr.extend(rr)

        # [TEST] :
        #break
    
    # compute unique & count
    unqiue_ss, count_ss = np.unique(all_ss, return_counts=True)
    unqiue_sr, count_sr = np.unique(all_sr, return_counts=True)
    unqiue_rs, count_rs = np.unique(all_rs, return_counts=True)
    unqiue_rr, count_rr = np.unique(all_rr, return_counts=True)

    # dicionary
    ss_stats = dict(zip(unqiue_ss, count_ss))
    sr_stats = dict(zip(unqiue_sr, count_sr))
    rs_stats = dict(zip(unqiue_rs, count_rs))
    rr_stats = dict(zip(unqiue_rr, count_rr))    

    return all_ss, all_sr, all_rs, all_rr, ss_stats, sr_stats, rs_stats, rr_stats 


def write_iat_to_text(ss_stats, sr_stats, rs_stats, rr_stats, file):

    with open(os.path.join(os.getcwd(), "stats", file+".txt"), "w") as file:
        # sent-to-sent:
        file.write(f"sent-to-sent:\n")
        for key, value in ss_stats.items(): 
            file.write(f"[{key}]: [{format(value, ',')}] \n")

        # sent-to-recv:
        file.write(f"\n\nsent-to-recv:\n")
        for key, value in sr_stats.items(): 
            file.write(f"[{key}]: [{format(value, ',')}] \n")

        # recv-to-sent:
        file.write(f"\n\nrecv-to-sent:\n")
        for key, value in rs_stats.items(): 
            file.write(f"[{key}]: [{format(value, ',')}] \n")

        # recv-to-recv:
        file.write(f"\n\nrecv-to-recv:\n")
        for key, value in rr_stats.items(): 
            file.write(f"[{key}]: [{format(value, ',')}] \n")        


def main():

    print(f"----------  [{os.path.basename(__file__)}]: start to run [{args['in']}]  ----------")

    # 1. load the pickle file:
    with open(os.path.join(os.getcwd(), "sim-traces", args["in"]), "rb") as f:
        dataset, iats, _ = pickle.load(f)

    print(f"[LOADED] [{args['in']}] pickle file")


    # 2. compute trace
    all_ss, all_sr, all_rs, all_rr, ss_stats, sr_stats, rs_stats, rr_stats = get_all_iats(dataset, iats)

    print(f"[GOT] all iats")


    # 3. (dump unique & count):
    with open(os.path.join(os.getcwd(), "stats", args["out"]+".pkl"), "wb") as f:
        pickle.dump((all_ss, all_sr, all_rs, all_rr, ss_stats, sr_stats, rs_stats, rr_stats), f)

        print(f"[SAVED] iats to the {args['out']}.pkl file")  


    # 4. write to the args["txt"] file 
    write_iat_to_text(ss_stats, sr_stats, rs_stats, rr_stats, args["out"])

    print(f"[SAVED] iats to the {args['out']}.txt file")   


    print(f"----------  [{os.path.basename(__file__)}]: complete successfully  ----------")    


if __name__ == "__main__":
    main()

