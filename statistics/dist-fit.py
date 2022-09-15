#!/usr/bin/env python3

"""
 distfit.py

"""

import argparse
import os
import pickle
import numpy as np
from scipy import stats
from distfit import distfit
import matplotlib.pyplot as plt 


# parse arguments: 
parser = argparse.ArgumentParser()

# 1. INPUT: 
parser.add_argument("--in", required=True, help="load dataset from pickle file")

# 2. OUTPUT: 
parser.add_argument("--out", required=True, help="save traces to the file.")

# 3. mode:
parser.add_argument("--mode", required=False, type=str, help="load dataset from pickle file")

# 4. max length
parser.add_argument("--index1", required=False, type=int, help="load dataset from pickle file")

# 5. max length
parser.add_argument("--index2", required=False, type=int, help="load dataset from pickle file")

args = vars(parser.parse_args())


# use distfit library:
def fit_data_to_dist(data):
    lines = []

    # Initialize model
    dist = distfit(distr=['uniform', 'logistic', 'fisk', 'exponweib', 'genpareto'])

    # Find best theoretical distribution for empirical data X
    res = dist.fit_transform(data)

    for key, value in res.items():
        lines.append(f"[{key}]: {value}\n\n")

    
    #dist.plot()
    #dist.plot_summary()


    return lines

# use sicpy fit() method
def fit_data_to_dist2(data):

    lines = []

    # [1] uniform distribution
    #uniform = stats.fit(stats.uniform, pos)
    uniform = stats.uniform.fit(data)
    lines.append(f"[uniform]: {uniform} \n")

    # [2] : logistic distribution
    #log = stats.fit(stats.logistic, pos)
    log = stats.logistic.fit(data)
    lines.append(f"[logistic]: {log} \n")

    # [3] : log-logistic distribution
    #fisk = stats.fit(stats.fisk, neg, [(0, 2), (20, 30)])
    fisk = stats.fisk.fit(data)
    lines.append(f"[log-logistic]: {fisk} \n")

    # [4] : geometirc distribution
    geom = stats.fit(stats.geom, data)
    lines.append(f"[geomeotric]: {geom} \n")

    #geom.plot()
    #plt.show()

    # [5] : weibull distribution
    #weibull = stats.fit(stats.exponweib, neg)
    weibull = stats.exponweib.fit(data)
    lines.append(f"[weibull]: {weibull} \n")

    # [6] : generalized pareto
    #pareto = stats.fit(stats.genpareto, pos)
    pareto = stats.genpareto.fit(data)
    lines.append(f"[generialized pareto]: {pareto} \n\n")  


    return lines  


def get_burst_distfit(file, distfit_func):

    # 1. load the pickle file: unique, count is np.ndarray
    with open(os.path.join(os.getcwd(), "stats", file), "rb") as f:
        unique, count, allbursts = pickle.load(f)

    print(f"[LOADED] {file} file")

    bursts= np.array(allbursts)

    send_burst = bursts[bursts> 0]
    recv_burst = -bursts[bursts< 0] 

    lines = []

    lines.append(f"send-burst = [{format(len(send_burst), ',')}], recv-burst = [{format(len(recv_burst), ',')}], all-bursts= [{format(len(bursts), ',')}] \n\n")

    lines.append(f"send burst fit to distribution:\n")
    lines.extend(distfit_func(send_burst))

    lines.append(f"recv burst fit to distribution:\n")
    lines.extend(distfit_func(recv_burst))


    return lines


def get_iat_distfit(file, index1, index2, distfit_func):

    # 1. load the pickle file:
    with open(os.path.join(os.getcwd(), "stats", file), "rb") as f:
        ss, sr, rs, rr, _, _, _, _ = pickle.load(f)

    print(f"[LOADED] {file} file")


    # 2. fit data to probability distribution :
    sent_to_sent = np.array(ss)
    recv_to_recv = np.array(rr)

    lines = []

    lines.append(f"sent-to-sent : {len(ss)}, sent-to-recv : {len(sr)}, \nrecv-to-sent : {len(rs)}, recv-to-recv : {len(rr)} \n\n")

    # sent-to-sent
    lines.append(f"sent-to-sent[:{index1}] iats fit to distribution:\n")
    print(f"[USED] sent-to-sent[:{index1}]")

    lines.extend(distfit_func(sent_to_sent[:index1]))

    # recv-to-recv
    lines.append(f"recv-to-recv[:{index2}] iats fit to distribution:\n")
    print(f"[USED] recv-to-recv[:{index2}]")

    lines.extend(distfit_func(recv_to_recv[:index2]))

    

    return lines


def main():
    
    print(f"----------  [{os.path.basename(__file__)}]: start to run, mode: [{args['mode']}]  ----------")


    # 1. burst:
    if args["mode"] == "burst":
        lines = get_burst_distfit(args["in"], fit_data_to_dist)
        file = os.path.join(os.getcwd(), "stats", "distfit", "burst-"+args["out"])
    # 2. iat:
    elif args["mode"] == "iat":
        lines = get_iat_distfit(args["in"], args["index1"], args["index2"], fit_data_to_dist)
        file = os.path.join(os.getcwd(), "stats", "distfit", "iat-"+args["out"])
    # 3. unrecognized mode
    else :
        sys.exit(f"[ERROR] unrecognized mode : [{args['mode']}]")


    # 4. write to *.txt file
    with open(file, "w") as f:
        f.writelines(lines)

        print(f"[SAVED] results to the {args['mode']+'-'+args['out']} file.")


    print(f"----------  [{os.path.basename(__file__)}]: complete successfully  ----------\n")    


if __name__ == "__main__":
    main()

