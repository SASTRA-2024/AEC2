#CORE CONFIGURATION FILE

#DATASET AND DATALOADERS 
annot_file = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\UrbanSound8K.csv"
audio_dir = "C:\\Users\\nagav\\OneDrive\\Documents\\UrbanSound8K.tar\\UrbanSound8K\\audio"
kfold = 10
num_test_folds = 1
drop_last = True 
batch_size = 32
#IMPORTS 
stmts = compile("""
#___________________________________________________________

#STANDARD IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import itertools as it
import sys 

#DEVICE AGNOSTIC CODE 

device = "cuda" if torch.cuda.is_available() else "cpu" 

#__________________________________________________________
""" , "" ,"exec")


exec(stmts) #USE THIS STATEMENT ACROSS THE MODULE TO EXECUTE COMMON STATEMENTS 

#____________________________STANDARD SIGNAL HYPER-PARAMETERS AND TRANSFORMS__________________________ 

sample_rate = 22050*4 
num_samples = 22050*4


#CAUTION : removing device agnostic for the transformations can affect various sections of the module
#(32 , 1 , 64 , 173)
melTransform = T.MelSpectrogram(
    sample_rate = sample_rate,
    n_fft = 1024  ,                             #frame length
    hop_length = 512 ,                          #half of the frame length
    n_mels = 64 
    ).to(device)
#(32 , 1 , 64 , 171)
mfccTransform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=64,
    melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 64, "center": False},
    ).to(device)
#(32 , 1 , 64 , 171)
lfccTransform = T.LFCC(
    sample_rate=sample_rate,
    n_lfcc=64,
    speckwargs={"n_fft": 1024, "hop_length": 512, "center": False},
    ).to(device)
   

#______________________________MODELS________________________________
#use class names specified in MyModels.py , this may affect other sections of the module
currModel = "CNN_Net"
epochs  = 1 
#valid spec options  "mfcc" , "lfcc" , "mel" , None (use None in ensumble model)
models = {
    "RNN_GRU" : {
                "params" : (171 , 32 , 8, 10),  #Class parameters
                "inDim" : (-1, 64, 171) ,       #input dim
                 "spec" : "mfcc" ,              #which spectrogram 
                 "toDB" : {"mfcc" : False} ,               #conversion to decible
                "loss_fun" : "nn.CrossEntropyLoss().to(device)", #loss functions 
                "optimizer" : "torch.optim.ASGD(model.parameters() , lr = 0.001)",#optimizers 
                 "path" : "C:\\Users\\nagav\\OneDrive\\Desktop\\ML_Moduled\\Trained\\RNN_GRU.pth", #model path
                 "results" : "\\performace\\RNN_GRU" #result path
                 },
    "CNN_Net" : {
                "params" : None ,  #Class parameters 
                "inDim" : (32 , 1 , 64 , 173) ,       #input dim
                 "spec" : "mel" ,              #which spectrogram 
                 "toDB" : {"mel" : True} ,               #conversion to decible
                "loss_fun" : "nn.CrossEntropyLoss().to(device)", #loss function 
                "optimizer" : "torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-07, weight_decay=1e-3)",#optimizers 
                 "path" : "C:\\Users\\nagav\\OneDrive\\Desktop\\ML_Moduled\\Trained\\CNN_Net.pth", #model path
                 "results" : "\\performace\\CNN_Net" #result path
                 },
    }

if __name__ == "__main__" :
    pass 

    