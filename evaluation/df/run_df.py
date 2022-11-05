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
CURRENT_TIME = time.strftime("%Y.%m.%d-c%H:%M:%S", time.localtime())

BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
INPUT_DIR = join(BASE_DIR, "simulation", "sim-traces")
OUTPUT_DIR = join(BASE_DIR, "results")
TRAINED_DF_DIR = join(BASE_DIR, "evaluation", "df", "model")

NONPADDING_SENT = 1.0
NONPADDING_RECV = -1.0


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
    CONFIG_FILE = join(os.getcwd(), "conf.ini")

    config_parser = configparser.ConfigParser()
    config_parser.read(CONFIG_FILE)

    return args, config_parser

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
    for trace in dataset:
        direct_trace = [row[1] for row in trace.tolist()]
        direct_trace = direct_trace + [0] * (5000 - len(direct_trace))
        X.append(np.array([direct_trace[:5000]], dtype=np.float32))

    for trace in X:
        trace[trace > 1.0] = 1.0
        trace[trace < -1.0] = -1.0    
        
    return X, y     


# load dataset
def spilt_dataset(X, y):
    # split to train & test [8:2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247, stratify=y)
    # split to validation & test [1:1]
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=247, stratify=y_test)

    train_data = DFDataset(X_train, y_train)
    valid_data = DFDataset(X_valid, y_valid)
    test_data = DFDataset(X_test, y_test)

    print(f"[SPLITED] traning size: {len(train_data)}, validation size: {len(valid_data)}, test size: {len(test_data)}")


    return train_data, valid_data, test_data
  

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
    print(f"[training] Avg_loss: {running_loss}/{num_batch} = {running_loss/num_batch}")        
 

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
        #
        for X, y in dataloader:
            # dataset load to device
            X, y = X.to(device), y.to(device)

            # 1. Compute prediction:
            pred = model(X)

            # 2. accumulate number of predicted corrections:
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # print accuracy:
    print(f" [validating] Accuracy: {correct}/{ds_size} = {correct/ds_size}")



def df_metrics(threshold, pred, label, label_unmon):
    # TP, FP-P, FP-N, TN, FN
    tp, fpp, fpn, tn, fn = 0, 0, 0, 0, 0

    # traverse preditions
    for i in range(len(pred)):
        
        # get prediction
        label_pred = np.argmax(pred[i])
        
        prob_pred = max(pred[i])
        
        label_correct = label[i]

        # we split on monitored or unmonitored correct label
        if label_correct != label_unmon:
            # either confident and correct,
            if prob_pred >= threshold and label_pred == label_correct:
                tp = tp + 1
            # confident and wrong monitored label, or
            elif prob_pred >= threshold and label_pred != label_unmon:
                fpp = fpp + 1
            # wrong because not confident or predicted unmonitored for monitored
            else:
                fn = fn + 1
        else:
            if prob_pred < threshold or label_pred == label_unmon: # correct prediction?
                tn = tn + 1
            elif label_pred < label_unmon: # predicted monitored for unmonitored
                fpn = fpn + 1
            else: # this should never happen
                sys.exit(f"this should never, wrongly labelled data for {label_pred}")

    # 
    # compute recall
    if tp + fn + fpp > 0:
        recall = round(float(tp) / float(tp + fpp + fn), 4)
  
    # compute precision      
    if tp + fpp + fpn > 0:
        precision = round(float(tp) / float(tp + fpp + fpn), 4)

    # compute F1
    if precision > 0 and recall > 0:
        f1 = round(2*((precision*recall)/(precision+recall)), 4)

    # compute accuracy
    accuracy = round(float(tp + tn) / float(tp + fpp + fpn + fn + tn), 4)

    
    return tp, fpp, fpn, tn, fn, accuracy, recall, precision, f1

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

        print(f" [testing] prediction: [{len(pred)}] , label: [{len(label)}]")
    
    # metrics result
    lines = []

    threshold = np.append([0], 1.0 - 1 / np.logspace(0.05, 2, num=15, endpoint=True))
    threshold = np.around(threshold, decimals=4)
    
    for th in threshold: 
        # compute metrics
        tp, fpp, fpn, tn, fn, accuracy, recall, precision, f1 = df_metrics(th, pred, label, classes)

        lines.append(f"[METRICS] TP: [{tp}] , FP-P: [{fpp}] , FP-N: [{fpn}] , TN: [{tn}] , FN: [{fn:>5}]\n")
        lines.append(f"[METRICS_1] threshold: [{th:4.2}], accuracy: [{accuracy:4.2}]\n")
        lines.append(f"[METRICS_2] precision: [{precision:4.2}] , recall: [{recall:4.2}] , F1: [{f1:4.2}]\n\n")

    return lines 

# [MAIN]
def main():
    logger = get_logger()
    # 
    logger.info(f"{MODULE_NAME}: start to run.")

    # parse commmand arguments & configuration file
    args, config = parse_arguments()

    EPOCH = int(config["default"]["epoch"])
    BATCH_SIZE = int(config["default"]["batch_size"])
    CLASSES = int(config["default"]["num_mon_site"])+1  # monitored + 1(unmonitored)

    # 1. load dataset
    X, y = preprocess_data(join(INPUT_DIR, args["in"]))
    train_data, valid_data, test_data = spilt_dataset(X, y)

    # select cpu/gpu mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    
    ############   TRAINING   ###############
    if args["train"]:
        #
        logger.info(f"----- [TRAINING] start to train the DF model -----")

        # train/validating dataloader: 
        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)

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
            # validate DF model
            validate_loop(valid_dataloader, df_net, device)

        logger.info(f"----- [TRAINING] Completed -----")

        # save trained DF model
        torch.save(df_net, join(TRAINED_DF_DIR, args["model"]))
        logger.info(f"[SAVED] the trained DF model to the {args['model']}")

    else: # load trained DF model
        df_net = torch.load(join(TRAINED_DF_DIR, args["model"])).to(device)
        logger.info(f"[LOADED] the trained DF model, file-name: {args['model']}")

    ############   TESTING   ################# 
    logger.info(f"----- [TESTING] start to test the DF model -----")

    # test dataloader:
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # run test
    lines = test(test_dataloader, df_net, device, CLASSES)
    
    # save testing results
    with open(join(OUTPUT_DIR, CURRENT_TIME+args["out"]+".txt"), "w") as f:
        f.writelines(lines)
        logger.info(f"[SAVED] testing results, file-name: {args['out']}")


    logger.info(f"{MODULE_NAME}: complete successfully.\n")


if __name__ == "__main__":
    sys.exit(main())