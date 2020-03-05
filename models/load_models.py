from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.losses import binary_crossentropy
from keras.activations import relu,elu,linear,sigmoid
from keras.optimizers import Adam,Nadam,RMSprop
from talos.model import lr_normalizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import talos as ta
from talos import Deploy

import operator
import time
from collections import namedtuple
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sortedcontainers import SortedDict

import glob

prefix = 'comp_edu'
num_splits = 15
for i in range(1,num_splits):
    dir_name = prefix + str(i)
    csv_file = glob.glob(dir_name + '/*')[0]
    print('Loading...',csv_file)
    df = pd.read_csv()


