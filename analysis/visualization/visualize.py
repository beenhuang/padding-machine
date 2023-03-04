#!/usr/bin/env python3

"""
<file>    visualize.py
<brief>   compute bandwidth overhead.
"""

import argparse
import os
import sys
import time
import pickle
import logging
import numpy as np
from PIL import Image
from os.path import join, basename, abspath, splitext, dirname, pardir, isdir


# CONSTANTS
MODULE_NAME = basename(__file__)
CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S_", time.localtime())

BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
INPUT_DIR = join(BASE_DIR, "simulation", "sim-traces")
OUTPUT_DIR = join(BASE_DIR, "analysis", "visualization", "result")

NONPADDING_SENT = 1.0
NONPADDING_RECV = -1.0
PADDING_SENT = 2.0
PADDING_RECV = -2.0

# cells of one trace
X_AXIS = 5000
# number of traces
Y_AXIS = 2000

# TOMATO colors below
# transparent PNG (alpha 0)
COLOR_BACKGROUND = [0, 0, 0, 0] 
# black - most data is nonpadding received
COLOR_NONPADDING_RECV = [0, 0, 0, 255] 
# white - sent nonpadding data
COLOR_NONPADDING_SENT = [255, 255, 255, 255] 
# red - most padding is received padding
COLOR_PADDING_RECV = [170, 57, 57, 255]
# green - outgoing padding 
COLOR_PADDING_SENT = [45, 136, 45, 255] 


#
def get_logger():
    logging.basicConfig(format="[%(asctime)s] >> %(message)s", level=logging.INFO)
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger


# [FUNC] parse arugment
def parse_arguments():
    # argument parser
    parser = argparse.ArgumentParser(description="visualization")

    # 1. INPUT: load ds-*.pkl dataset
    parser.add_argument("-i", "--in", required=True, metavar="<simulated-traces>", help="load simulated traces from pickle file")
    # 2. OUTPUT: save overhead in the overhead-*.txt file
    parser.add_argument("-o", "--out", required=True, metavar="<result-file>", help="save overhead to the text file")

    args = vars(parser.parse_args())

    return args


#
def get_img_data(dataset, n, width):
    data = np.full((n, width, 4), COLOR_BACKGROUND, dtype=np.uint8)

    # draw 1,000 traces
    for y, trace in enumerate(dataset):
        print(f"number of trace: {y}")
        if y >= n:
            break

        x = 0
        for cell in trace[:,1]:
            #print(f"trace: {trace}")
            if x >= width:
                break

            if cell == NONPADDING_SENT:
                data[y][x] = COLOR_NONPADDING_SENT
            elif cell == NONPADDING_RECV:
                data[y][x] = COLOR_NONPADDING_RECV
            elif cell == PADDING_SENT:
                data[y][x] = COLOR_PADDING_SENT
            elif cell == PADDING_RECV:
                data[y][x] = COLOR_PADDING_RECV
            
            x += 1

    return data


# MAIN FUNCTION
def main():
    logger = get_logger()
    logger.info(f"{MODULE_NAME}: start to run.")

    # parse arguments
    args = parse_arguments()
    logger.info(f"Arguments: {args}")

    # [1] load dataset & label:
    with open(join(INPUT_DIR, args["in"]), "rb") as f:
        dataset, _ = pickle.load(f)
    logger.info(f"[LOADED] dataset from the {args['in']}")    

    # [2] get image data:
    image = Image.fromarray(get_img_data(dataset, Y_AXIS, X_AXIS))
    logger.info(f"[GOT] image data.")

    # make "result" directory
    if not exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # [3] write bandwidth overhead to the file.
    with open(join(OUTPUT_DIR, CURRENT_TIME+args["out"]+".png"), "wb") as f:
        image.save(f)
        logger.info(f"[SAVED] image data in the {args['out']}.png")

    logger.info(f"{MODULE_NAME}: complete successfully.\n")
        

if __name__ == "__main__":
    sys.exit(main())

