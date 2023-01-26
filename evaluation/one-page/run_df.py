#!/usr/bin/env python3

"""
<file>    df.py
<brief>   train&test DF model
"""

import argparse
import configparser
import os 
import sys       
import csv
import time
import pickle
import logging
import numpy as np
from os.path import join, basename, abspath, dirname, pardir, splitext

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adamax
import torch.nn.functional as F
import torch.nn as nn

from dfnet import DFNet


# constants: module-name, file-path, batch-size
MODULE_NAME = basename(__file__)
CURRENT_TIME = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
INPUT_DIR = join(BASE_DIR, "simulation", "sim-traces")
OUTPUT_DIR = join(BASE_DIR, "results")
CONFIG_DIR = join(BASE_DIR, "evaluation", "df")
TRAINED_DF_DIR = join(BASE_DIR, "evaluation", "df", "trained-model")

NONPADDING_SENT = 1.0
NONPADDING_RECV = -1.0

PADDING_SENT = 2.0
PADDING_RECV = -2.0


#
def get_logger():
    logging.basicConfig(format="[%(asctime)s] >> %(message)s", level=logging.INFO)
    logger = logging.getLogger(splitext(basename(__file__))[0])
    
    return logger

# parse argument 
def parse_arguments():
    # [1] argument parser
    parser = argparse.ArgumentParser(description="Deep Fingerprinting")
    
    # INPUT: load dataset.
    parser.add_argument("-i", "--in", required=True, 
                        help="load dataset&labels.")
    # OUTPUT: save test resulting.
    parser.add_argument("-o", "--out", required=True, 
                        help="save test results.")
    # TRAINING: do run training
    parser.add_argument("--train", required=False, action="store_true", 
                        default=False, help="do run training DF model.")
    # TRAINING_output: save trained DF model.
    parser.add_argument("-m", "--model", required=False, 
                        help="save trained DF model.")

    args = vars(parser.parse_args())

    # [2] configuration parser
    config_file = join(CONFIG_DIR, "conf.ini")

    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)

    return args, config_parser["default"]

class DFDataset(Dataset):
    def __init__(self, X, y):
        self.datas=X
        self.labels=y
        
    def __getitem__(self, index):
        data = torch.from_numpy(self.datas[index]) 
        label = torch.tensor(self.labels[index]) 

        return data, label
    
    def __len__(self):
        return len(self.datas)

def preprocess_data(file):
    #
    with open(file, "rb") as f:
        dataset, y = pickle.load(f)

    X = []
    for elem in dataset:
        direct_trace = [row[1] for row in elem.tolist()]
        direct_trace = direct_trace + [0] * (5000 - len(direct_trace))
        trace = np.array([direct_trace[:5000]], dtype=np.float32)
        trace[trace == PADDING_SENT] = NONPADDING_SENT
        trace[trace == PADDING_RECV] = NONPADDING_RECV    
        X.append(trace)
    

    return X, y     


# load dataset
def spilt_dataset(X, y):
    # split to train & test [8:2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247, stratify=y)
 
    train_data = DFDataset(X_train, y_train)
    test_data = DFDataset(X_test, y_test)

    print(f"[SPLITED] traning size: {len(train_data)}, test size: {len(test_data)}")

    return train_data, test_data
  

# dataloader : training=16,000 , batch_size=750, num_batch=22
def train_loop(dataloader, model, loss_function, optimizer, device):
    # loss value
    running_loss = 0.0
    # number of batches 16,000/750=22
    num_batch = len(dataloader)

    # update batch_normalization and enable dropout layer
    model.train()
    
    # loop
    for X, y in dataloader:
        # dataset load to device
        X, y = X.to(device), y.to(device)

        # 1. Compute prediction error
        pred = model(X)
        loss = loss_function(pred, y)

        # 2. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate loss value
        running_loss += loss.item()
    
    # print loss average:
    print(f"[training] Avg_loss(loss/num_batch): {running_loss}/{num_batch} = {running_loss/num_batch}")        
 

# dataloader : dataset=2,000 , batch_size=750, num_batch=3
def validate_loop(dataloader, model, device):
    # number of prediction correct
    correct = 0.0
    # number of valiation samples: 2000
    ds_size = len(dataloader.dataset)

    # not update batch_normalization and disable dropout_layer
    model.eval()

    # set gradient calculation to off
    with torch.no_grad():
        for X, y in dataloader:
            # dataset load to device
            X, y = X.to(device), y.to(device)

            # 1. Compute prediction:
            pred = model(X)

            # 2. accumulate number of predicted corrections:
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # print accuracy:
    print(f"[validating] Accuracy: {correct}/{ds_size} = {correct/ds_size}")


# [FUNC]: test DF model
def test(dataloader, model, device, classes):
    # prediction, true label
    pred, label = [], []
    
    # not update batch_normalization and disable dropout layer
    model.eval()

    # set gradient calculation to off
    with torch.no_grad():
        # travel
        for X, y in dataloader:
            # dataset load to device
            X, y = X.to(device), y.to(device)

            # 1. Compute prediction:
            prediction = model(X)

            # extend softmax result to the prediction list:
            pred.extend(F.softmax(prediction, dim=1).data.cpu().numpy().tolist())

            # extend actual label to the label list:
            label.extend(y.data.cpu().numpy().tolist())

        print(f"[testing] prediction: {len(pred)}, label: {len(label)}")
    
    lines, tpr, fpr = get_openworld_score(label, pred, max(label))

    return lines, tpr, fpr 

# get binary classification score
def get_binary_score(y_true, y_pred, label_unmon):
    # TP, FN  TN, FN
    tp, fn, tn, fp = 0, 0, 0, 0

    # traverse preditions
    for i in range(len(y_pred)):
        # [case_1]: positive sample, and predict positive and correct.
        if y_true[i] != label_unmon and y_pred[i] == y_true[i]:
            tp += 1
        # [case_3]: positive sample, predict negative.
        elif y_true[i] != label_unmon and y_pred[i] != y_true[i]:
            fn += 1
        # [case_4]: negative sample, predict negative.    
        elif y_true[i] == label_unmon and y_pred[i] == y_true[i]:
            tn += 1
        # [case_5]: negative sample, predict positive    
        elif y_true[i] == label_unmon and y_pred[i] != y_true[i]:
            fp += 1   
        else:
            sys.exit(f"[ERROR]: {y_pred[i]}, {y_true[i]}")        

    # accuracy
    accuracy = (tp+tn) / float(tp+fn+tn+fp)
    # precision      
    precision = tp / float(tp+fp)
    # recall
    recall = tp / float(tp+fn)
    # F-score
    f1 = 2*(precision*recall) / float(precision+recall)
    # FPR
    fpr = fp / float(fp+tn)

    lines = []
    lines.append(f"[POS] TP: {tp},  FN: {fn}\n")
    lines.append(f"[NEG] TN: {tn},  FP: {fp}\n\n")
    lines.append(f"accuracy: {accuracy}\n")
    lines.append(f"precision: {precision}\n")
    lines.append(f"recall (TPR): {recall}\n")
    lines.append(f"F1: {f1}\n")
    lines.append(f"FPR: {fpr}\n\n\n")
    
    return lines, recall, fpr

def choose_one_mon_class(data, labels, mon_label, unmon_label):
    X, y = [], []
    MAX_INSTANCE = 100000
    n = 0

    for index, label in enumerate(labels):
        if mon_label == label:
            X.append(data[index])
            y.append(labels[index])
        
        if unmon_label == label and n < MAX_INSTANCE:
            n += 1
            X.append(data[index])
            y.append(labels[index])

    return X, y  

# main function
def main(input, output, epoch, batch_size, logger):
    EPOCH = epoch
    BATCH_SIZE = batch_size
    CLASSES = 2 # 1-monitored + 1-unmonitored

    # 1. load dataset
    infile = join(INPUT_DIR, input)
    data, labels = preprocess_data(infile)
    
    # select cpu/gpu mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    tpr, fpr, n = 0.0, 0.0, 0
    # loop: select one monitored class with one unmonitored class
    for mon_label in range(max(labels)):
        n += 1
        print(f"mon_label: {mon_label}")
        X, y = choose_one_mon_class(data, labels, mon_label, max(labels))
        logger.info(f"[GOT] X_length:{len(X)}, y_length:{len(y)}, labels:{list(set(y))}")

        # split data
        train_data, test_data = spilt_dataset(X, y)
    
        # train
        logger.info(f"----- [TRAINING] start to train the DF model -----")

        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
   
        # create DFNet model    
        df_net = DFNet(CLASSES).to(device)         
        # loss function:
        loss_function = nn.CrossEntropyLoss()
        # optimizer:
        optimizer = Adamax(params=df_net.parameters())

        # training loop
        for i in range(EPOCH):
            logger.info(f"---------- Epoch {i+1} ----------")

            # train DF model
            train_loop(train_dataloader, df_net, loss_function, optimizer, device)
      
        logger.info(f"----- [TRAINING] Completed -----")


        # test
        logger.info(f"----- [TESTING] start to test the DF model -----")

        # test dataloader:
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

        # run test
        lines, tmp_tpr, tmp_fpr = test(test_dataloader, df_net, device, CLASSES)
        tpr += tmp_tpr
        fpr += tmp_fpr
        logger.info(f"[CALCULATED] metrics, TPR:{tmp_tpr}, FPR:{tmp_fpr}.")
   
        # save testing results
        outfile = join(OUTPUT_DIR, output+".txt")
        with open(outfile, "a") as f:
            f.writelines(lines)
            logger.info(f"[SAVED] testing results, file-name: {args['out']}")

    logger.info(f"TPR:{tpr/n}, FPR:{fpr/n}, num_loop:{n}")
    logger.info(f"{MODULE_NAME}: complete successfully.\n")


if __name__ == "__main__":
    try:
        logger = get_logger()
        logger.info(f"{MODULE_NAME}: start to run.")

        # parse commmand arguments & configuration file
        args, config = parse_arguments()
        logger.info(f"args: {args}, config: {config}")

        main(args["in"], args["out"], int(config["epoch"]), int(config["batch_size"])logger=logger)

    except KeyboardInterrupt:
        sys.exit(1) 
