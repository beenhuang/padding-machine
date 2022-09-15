#!/usr/bin/env python3

"""

  df.py

"""

import argparse
import os 
import sys       
import pickle
import csv

import numpy as np
import torch
from torch.utils.data import Dataset     
from torch.utils.data import DataLoader  
import torch.nn as nn  
import torch.nn.functional as F 

#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# parse arguments: 
parser = argparse.ArgumentParser()

# INPUT & OUTPUT
# [1] INPUT: load dataset from ds-*.pkl
parser.add_argument("--ld", required=True, 
                    help="load dataset from the pickle file which is original dataset or a simulation result")

# [2] OUTPUT : save resulting metrics in the res-*.pkl file
parser.add_argument("--csv", required=False, 
                    default=None, help="save resulting metrics in provided path in csv format")

# TRAINING
# TRAINING : do run training
parser.add_argument("--train", required=False, action="store_true", 
                    default=False, help="do run training DF model")

# TRAINING_parameter: number of loop
parser.add_argument("--epoch", required=False, type=int, 
                    default=30, help="the number of epoch for training")

# TRAINING/TESTING_parameter: batch size
parser.add_argument("--batchsize", required=False, type=int, 
                    default=750, help="batch size")

# TRAINING_output: save trained DF model to df-*.pkl
parser.add_argument("--sm", required=False, 
                    default="", help="save model to the pickle file")

# TESTING
# TESTING_input : load the trained DF model from df-*.pkl
parser.add_argument("--lm", required=False, 
                    default="", help="load model from pickle file")

# DATASET
# DATASET_parameter: class:partition:sample; partition offset
parser.add_argument("--class", required=False, type=int, 
                    default=50, help="the number of monitored classes")
parser.add_argument("--part", required=False, type=int, 
                    default=10, help="the number of monitored parts")
parser.add_argument("--sample", required=False, type=int, 
                    default=20, help="the number of monitored samples")
parser.add_argument("--fold", required=False, type=int, 
                    default=0, help="the fold number (partition offset)")

# WHY?? DATASET_parameter: zero sample[start:stop]
parser.add_argument("--zero", required=False, 
                    default="", help="zero each sample between sample[zero], e.g., 0:10 for the first 10 cells")

# args                    
args = vars(parser.parse_args())


'''

  zero [start:stop]

'''
def zero_dataset(dataset, zero):

    # start & stop index:
    start = int(zero.split(":")[0])
    stop = int(zero.split(":")[1])

    # data initilaize
    data = np.zeros((stop-start), dtype=np.float32)

    # zero dataset[start:stop]
    for ID, cellseq in dataset.items():
        cellseq[:,start:stop] = data
        dataset[ID] = cellseq

    return dataset

'''

  Splits the dataset based on fold.

  The split is only based on IDs, not the actual data. The result is a 8:1:1
  split into training, valid, and testing.

  # train=50*8*20 , validation=50*1*20 , test=50*1*20

  # fold=0, valid=8, test=9
  # fold=2, valid=6, test=7

'''
def split_dataset(classes, parts, samples, label, fold):
    
    # train/valid/test ID_list:
    train = []
    valid = []
    test = []

    # split dictionary: 'train':ID_list, 'valid':ID_list, 'test':ID_list
    split = {}

    # 1. monitored

    # traverse classes
    for cls in range(0,classes):
        # traverse partitions:
        for part in range(0,parts):
            # index:
            idx = (part + fold) % parts

            # traverse samples:
            for smpl in range(0,samples):

                # 1. ID:
                ID = f"m-{cls}-{part}-{smpl}"
                
                # add to training list
                if idx < parts-2:
                    train.append(ID)
                # add to valid list    
                elif idx < parts-1:
                    valid.append(ID)
                # add to testing list    
                else:
                    test.append(ID)

    # 2. unmonitored:

    # counter:
    counter = 0

    # key is ID
    for ID in label.keys():

        # skip not 'u'
        if not ID.startswith("u"):
            continue

        n = (counter+fold) % parts
        
        # train dataset
        if n < parts-2:
            train.append(ID)
        # valiation dateset    
        elif n < parts-1:
            valid.append(ID)
        # test dataset    
        else:
            test.append(ID) 

        # add 1 to the counter:
        counter += 1
    
    # add to the split dictionary
    split["train"] = train
    split["valid"] = valid
    split["test"] = test

    return split

'''

  DF dataset:
    instance variables: id, dataset, label

'''
class DF_Dataset(Dataset):
    
   def __init__(self, id, dataset, label):

       # ID: ID list
       self.id = id
       
       # dataset: ID:cell trace
       self.dataset = dataset

       # label: ID:class
       self.label = label

   def __len__(self):

       # reuturn the number of the dataset: 
       return len(self.id)

   def __getitem__(self, idx):

       # get ID:
       ID = self.id[idx]
       
       # return dataset/label coressponding the ID:
       return self.dataset[ID], self.label[ID]

'''
       
  DF model
       
'''    
           
class DFNet(nn.Module):
    def __init__(self, classes):
        super(DFNet, self).__init__()

        # full connection layer input features.
        self.fc_in_features = 512*10

        # convolution layer's kernel_size
        self.kernel_size = 7

        # convlutional/maxpool layer's padding_size
        self.padding_size = 3

        # maxpool's stride_size
        self.pool_stride_size = 4

        # maxpool's pool_size
        self.pool_kernel_size = 7


        self.block1 = self.__block(1, 32, nn.ELU())
        self.block2 = self.__block(32, 64, nn.ReLU())
        self.block3 = self.__block(64, 128, nn.ReLU())
        self.block4 = self.__block(128, 256, nn.ReLU())

        self.fc = nn.Sequential(
            # in_features, out_features
            nn.Linear(in_features=self.fc_in_features, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.prediction = nn.Sequential(
            nn.Linear(512, classes),
            # when using CrossEntropyLoss, already computed internally
            #nn.Softmax(dim=1) # dim = 1, don't softmax batch
        )
    
    def __block(self, channels_in, channels, activation):
        return nn.Sequential(
            # in_channels, out_channels, kernel_size, padding_size
            nn.Conv1d(in_channels=channels_in, out_channels=channels, kernel_size=self.kernel_size, padding=self.padding_size),
            # number of features
            nn.BatchNorm1d(num_features=channels),
            # non-linear change
            activation,

            nn.Conv1d(channels, channels, kernel_size=self.kernel_size, padding=self.padding_size),
            nn.BatchNorm1d(channels),
            activation,

            # kernel_size, stride, padding 
            nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride_size, padding=self.padding_size),
            # probability
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(start_dim=1) # dim = 1, don't flatten batch
        x = self.fc(x)
        x = self.prediction(x)

        return x    

'''

  training loop of the DF model

  dataloader : training_dataset=16,000 , batch_size=750, num_batch=22

  print loss average 

'''
def train_loop(dataloader, model, loss_fn, optimizer, device):

    # loss value
    running_loss = 0.0

    # number of batches 16,000/750=22
    num_batch = len(dataloader)

    # update batch_normalization and enable dropout layer
    model.train()

    # set gradient calculation to on 
    #torch.set_grad_enabled(True)
    
    # loop batch_size(default=750) times:
    for X, y in dataloader:
        
        # dataset load to device
        X, y = X.to(device), y.to(device)

        # 1. Compute prediction error:
        pred = model(X)
        loss = loss_fn(pred, y)

        # 2. Backpropagation:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate loss value
        running_loss += loss.item()
    
    # print loss average:
    print(f" [training] Avg loss: {running_loss}/{num_batch} = {running_loss/num_batch}")        
    
'''

  validating loop of DF model

  dataloader : dataset_size=2,000 , batch_size=750, num_batch=3

  print accuracy

'''
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
    print(f" [validating] Accuracy: {correct}/{ds_size} = {correct/ds_size}")

''' 

  Computes a range of metrics.

  For details on the metrics, see, e.g., https://www.cs.kau.se/pulls/hot/baserate/

'''

def df_metrics(threshold, pred, label, label_unmon):

    # TP, FP-P, FP-N, TN, FN
    tp, fpp, fpn, tn, fn = 0, 0, 0, 0, 0

    #mon, umn, total = 0

    # accuracy, recall, precision, F1
    accuracy, recall, precision, f1 = 0.0, 0.0, 0.0, 0.0

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
 
'''

  testing DF model 

'''
def test(dataloader, model, device):

    # prediction
    pred = []

    # actual label
    label = []
    
    # not update batch_normalization and disable dropout layer
    model.eval()

    # set gradient calculation to off
    with torch.no_grad():
        # travers dataloader
        for X, y in dataloader:

            # dataset load to device
            X, y = X.to(device), y.to(device)

            # 1. Compute prediction:
            prediction = model(X)

            # extend softmax result to the prediction list:
            pred.extend(F.softmax(prediction, dim=1).data.cpu().numpy().tolist())

            # extend actual label to the label list:
            label.extend(y.data.numpy().tolist())

        print(f" [testing] prediction: [{len(pred)}] , label: [{len(label)}]")
    
    # metrics result
    metrics = []

    threshold = np.append([0], 1.0 - 1 / np.logspace(0.05, 2, num=15, endpoint=True))
    threshold = np.around(threshold, decimals=4)
    
    for th in threshold:
        
        # compute metrics
        tp, fpp, fpn, tn, fn, accuracy, recall, precision, f1 = df_metrics(th, pred, label, args["class"])

        # append item to metrics list
        metrics.append([th, recall, precision, f1, tp, fpp, fpn, tn, fn])
        
        print(f" [METRICS1] TP: [{tp}] , FP-P: [{fpp}] , FP-N: [{fpn}] , TN: [{tn}] , FN: [{fn:>5}]")
        print(f" [METRICS2] threshold: [{th:4.2}], accuracy: [{accuracy:4.2}]")
        print(f" [METRICS3] precision: [{precision:4.2}] , recall: [{recall:4.2}] , F1: [{f1:4.2}]")

    return metrics              

def main():

    print(f"-------  [{os.path.basename(__file__)}]: start to run DF model, input: {args['ld']}  -------")
    #os.environ['KMP_DUPLICATE_LIB_OK']='True'

    # 1. load dataset & label

    with open(os.path.join(os.getcwd(), "sim-traces", args["ld"]), "rb") as f:
        dataset, label = pickle.load(f)

    # flatten dataset with extra details
    for k in dataset:
        dataset[k][0][dataset[k][0] > 1.0] = 1.0
        dataset[k][0][dataset[k][0] < -1.0] = -1.0            

    # loaded dataset & label
    print(f"[LOADED] dataset: [{len(dataset)}] , label: [{len(label)}] ")


    # split dataset: train:ID_list, valid:ID_list, test:ID_list
    split = split_dataset(args["class"], args["part"], args["sample"], label, args["fold"])
    
    # generate train/valid/test dataset:
    train_data = DF_Dataset(split["train"], dataset, label)
    valid_data = DF_Dataset(split["valid"], dataset, label)
    test_data = DF_Dataset(split["test"], dataset, label)

    print(f"[SPLITED] traning: [{len(split['train'])}] =50*8*20 , validating: [{len(split['valid'])}] =50*1*20 , testing: [{len(split['test'])}] =50*1*20 ")


    # zero sample[start:stop] 
    if args["zero"] != "":
        # zero sample[start:stop]
        dataset = zero_dataset(dataset, args["zero"])

        print(f"[ZERO] each item in dataset as data[{args['zero']}]")


    # select cpu/gpu mode:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"[USED]: [{device}] mode")
    

    # 2. training:
    if args["train"]:

        print(f"----- [TRAINING] start to train the DF model -----")

        # train dataloader: 
        train_dataloader = DataLoader(train_data, batch_size=args["batchsize"], shuffle=True)
        
        # valid dataloader:
        valid_dataloader = DataLoader(valid_data, batch_size=args["batchsize"], shuffle=True)

        # create new nn model on the device:    
        model = DFNet(args["class"]+1).to(device) 

        # loss function:
        loss_fn = nn.CrossEntropyLoss()

        # optimizer:
        optimizer = torch.optim.Adamax(params=model.parameters())

        # training loop
        for t in range(args["epoch"]):

            print(f"---------- Epoch {t+1} ----------")
            
            # train DF model
            train_loop(train_dataloader, model, loss_fn, optimizer, device)

            # validate DF model
            validate_loop(valid_dataloader, model, device)

        print(f"----- [TRAINING] Completed -----")

        
        # save trained DF model:
        if args["sm"] != "":
            # save trained DF model
            torch.save(model, os.path.join(os.getcwd(), "results", "model", args["sm"]))
            
            print(f"[SAVED] the trained DF model to the [{args['sm']}] file")
    
    # 3. testing: 
    
    print(f"----- [TESTING] start to test the DF model -----")

    # load trained DF model from the pickle file:
    if args["lm"] != "":
        # load trained DF model
        model = torch.load(os.path.join(os.getcwd(), "results", "model", args["lm"])).to(device)
        
        print(f'[LOADED] the trained [{args["lm"]}] DF model')

    # test dataloader:
    test_dataloader = DataLoader(test_data, batch_size=args["batchsize"])

    # run test
    metrics = test(test_dataloader, model, device)
    
    
    # saved metrics results to the [res-*.csv] csv file:
    if args["csv"]:
        with open(os.path.join(os.getcwd(), "results", args["csv"]), "w", newline="") as f:
            # create writer
            writer = csv.writer(f, delimiter=",")

            # write column name
            writer.writerow(["THRESHOLD", "RECALL", "PRECISION", "F1", "FP", "FP-P", "FP-N", "TN", "FN"])
           
            # write metrics result 
            writer.writerows(metrics)

        print(f"[SAVED] metrics result to [{args['csv']}] ")


    print(f"-------  [{os.path.basename(__file__)}]: complete successfully  -------\n")


if __name__ == "__main__":
    main()
