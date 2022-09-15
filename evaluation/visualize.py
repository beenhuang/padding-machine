#!/usr/bin/env python3

"""

visualize.py

"""

import argparse
import pickle

import numpy as np
from PIL import Image

# parse arguments:
parser = argparse.ArgumentParser()

# input and output file.
parser.add_argument("-i", required=True, help="load dataset from pickle file, provide the filepath.")
parser.add_argument("-o", default="test", help="save filename prefix.")

# dimensions of the image.
parser.add_argument("-x", type=int, default=5000, help="image width (x-axis)")
parser.add_argument("-y", type=int, default=1000, help="image height (y-axis)")

# hide 
parser.add_argument("--hide", required=False, default=False, action="store_true", help="hide padding cells")
                    
args = parser.parse_args()


# TOMATO colors below
COLOR_BACKGROUND = [0, 0, 0, 0] # transparent PNG (alpha 0)
COLOR_NONPADDING_RECV = [0, 0, 0, 255] # black - most data is nonpadding received
COLOR_NONPADDING_SENT = [255, 255, 255, 255] # white - sent nonpadding data
COLOR_PADDING_RECV = [170, 57, 57, 255] # red - most padding is received padding
COLOR_PADDING_SENT = [45, 136, 45, 255] # green - outgoing padding

def get_img_data(dataset, n, width):
    data = np.full((n, width, 4), COLOR_BACKGROUND, dtype=np.uint8)

    for y, k in enumerate(dataset):
        if y >= n:
            break
        x = 0
        for v in dataset[k][0]:
            if x >= width:
                break
            if v == 1:
                data[y][x] =COLOR_NONPADDING_SENT
                x += 1
            elif v == -1:
                data[y][x] = COLOR_NONPADDING_RECV
                x += 1
            elif not args.hide and v == 2:
                data[y][x] = COLOR_PADDING_SENT
                x += 1
            elif not args.hide and v == -2:
                data[y][x] = COLOR_PADDING_RECV
                x += 1

    return data

def main():
    # start to run the program:
    print(f"<-----   [visualize.py]: start to run   ----->")
    
    # 1. load pickle file:
    with open(args.i, "rb") as pf:
        dataset, _ = pickle.load(pf)

    print(f"1. loaded pickle dataset from pickle file [{args.i}].")    

    # 2. get image data:
    image = Image.fromarray(get_img_data(dataset, args.y, args.x))

    print(f"2. geted image data.")    
    
    # 3. save image data:
    with open(f"{args.o}.png", "wb") as f:
        image.save(f)
        
    print(f"3. saved image data in the [{args.o}.png]. ") 

    # end of the program:
    print(f"<-----   [visualize.py]: complete successfully   ----->")    


if __name__ == "__main__":
    main()

    