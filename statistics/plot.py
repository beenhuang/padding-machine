#!/usr/bin/env python3

"""
plot.py

"""

import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


# parse arguments: 
parser = argparse.ArgumentParser()

# 1. INPUT: 
parser.add_argument("--in", required=True, help="load dataset from pickle file")

# 2. OUTPUT: 
parser.add_argument("--out", required=True, help="load dataset from pickle file")

# 3. mode:
parser.add_argument("--mode", required=False, type=str, help="load dataset from pickle file")

# 4. max length
parser.add_argument("--index1", required=False, type=int, help="load dataset from pickle file")

# 5. max length
parser.add_argument("--index2", required=False, type=int, help="load dataset from pickle file")

args = vars(parser.parse_args())


#
def save_bar_plot(x, y, file):

    plt.bar(x, y)
    #plt.show()

    plt.savefig(os.path.join(os.getcwd(), "stats", "plot", file))

    plt.close()

#
def get_burst_plot(in_file, out_file, min, max):
    # 1. load the pickle file: unique, count is np.ndarray
    with open(os.path.join(os.getcwd(), "stats", in_file), "rb") as f:
        unique, count, _ = pickle.load(f)

    print(f"[LOADED] {in_file} file")


    # index of burst == 1
    index = int(np.argwhere(unique == 1))

    # 2. send burst
    send_burst = unique[index:index+max]
    send_count = count[index:index+max]

    save_bar_plot(send_burst, send_count, "send-"+out_file)


    # 3. recv burst
    recv_burst = unique[index-min:index]
    recv_count = count[index-min:index]

    save_bar_plot(recv_burst, recv_count, "recv-"+out_file)        


    print(f"[SAVED] bar plot to the {out_file} file")


#
def get_iat_plot(in_file, out_file, index1, index2):
    # 1. load the pickle file:
    # unique, count is np.ndarray
    with open(os.path.join(os.getcwd(), "stats", in_file), "rb") as f:
        _, _, _, _, ss_stats, sr_stats, rs_stats, rr_stats = pickle.load(f)

    print(f"[LOADED] {in_file} file")

    # send-to-send plot
    ss_iat = list(ss_stats.keys())
    ss_count = list(ss_stats.values())

    print(f"[USED] sent-to-sent max value: ss_*[:{index1}]")

    save_bar_plot(ss_iat[:index1], ss_count[:index1], "ss-"+out_file)



    # recv-to-recv plot
    rr_iat = list(rr_stats.keys())
    rr_count = list(rr_stats.values())

    print(f"[USED] recv-to-recv max value: rr_*[:{index2}]")

    save_bar_plot(rr_iat[:index2], rr_count[:index2], "rr-"+out_file)


    print(f"[SAVED] bar plot to the {out_file} file")



def main():

    print(f"----------  [{os.path.basename(__file__)}]: start to run, mode: [{args['mode']}]  ----------")


    if args["mode"] == "burst":
        get_burst_plot(args["in"], args["out"], args["index1"], args["index2"])

    elif args["mode"] == "iat":
        get_iat_plot(args["in"], args["out"], args["index1"], args["index2"])

    else :
        sys.exit(f"[ERROR] unrecognized mode : [{args['mode']}]")


    print(f"----------  [{os.path.basename(__file__)}]: complete successfully  ----------\n")    


if __name__ == "__main__":
    main()

