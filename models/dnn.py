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


#last layer neuron number
label_len=1
#grid_downsample for random search
grid_downsample=0.01

# parameters for talos
p = {'lr': (0.05, 0.5, 5),
     'neuron_tuple':[[50,25,20],[25,20],[20]],
     'batch_size': (2, 10, 30),
     'epochs': [100,200],
     'dropout': (0, 0.5, 5),
     'optimizer': [Adam, Nadam],
     'losses': [binary_crossentropy],
     'activation':[relu, elu],
     'last_activation': [sigmoid]}

def main():
    THRESHOLD = 0.75
    x_dataset,y_dataset=load_data()
    y_dataset = 1.0 * (y_dataset>=THRESHOLD)
    # get training dataset
    x_dataset,x_valid,y_dataset,y_valid=train_test_split(x_dataset,y_dataset,test_size=0.30)
    x_valid,x_test,y_valid,y_test=train_test_split(x_valid,y_valid,test_size=0.50)
    # join dataset for sorting
    z_dataset = np.concatenate((x_dataset,np.reshape(y_dataset,(y_dataset.shape[0],1))),axis=1)
    z_dataset = np.sort(z_dataset,axis=0)
    x_dataset,y_dataset=z_dataset[:,:-1],z_dataset[:,-1]

    np.save(open('test_setx.np','wb'),x_test)
    np.save(open('test_sety.np','wb'),y_test)
    np.save(open('valid_setx.np','wb'),x_test)
    np.save(open('valid_sety.np','wb'),y_test)
    # For each split, train
    NUM_SPLIT = 15
    for i in range(NUM_SPLIT):
        ratio = float(i+1) / float(NUM_SPLIT)
        x_train,y_train=filter_data(x_dataset,y_dataset,ratio)
        if x_train.shape[0] == 0:
            continue
        t=ta.Scan(x=x_train,
                  y=y_train,
                  model=fp_model,
                  fraction_limit=grid_downsample,
                  params=p,
                  experiment_name=('comp_edu'+str(i)),
                  x_val=x_valid,
                  y_val=y_valid)
        # make unique model name
        model_name = 'dnn_' + str(i) + '_' + str(NUM_SPLIT) + '.md'
        Deploy(scan_object=t,model_name=model_name,metric='val_loss',asc=True)
        #pickle.dump(t,open(model_name,'wb'))

def filter_data(x,y,ratio):
    # number of datapoints with (day ratio) <= ratio
    count = np.sum(x[:,0]<=ratio)
    return x[:count,:],y[:count]


def fp_model(x_train,y_train,x_val,y_val,params):
    # To train
    model=Sequential()
    layers=params['neuron_tuple']
    num_h=len(layers)
    for i in range(num_h):
        num_neurons=layers[i]
        model.add(Dense(num_neurons,activation=params['activation']))
        model.add(Dropout(params['dropout']))
    model.add(Dense(1,activation=params['last_activation']))
    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],params['optimizer'])),
                  loss=params['losses'],
                  metrics=['accuracy'])

    #Train the model, iterating on the data in batches of 32 samples
    history=model.fit(x_train,y_train,
              epochs=params['epochs'],
              batch_size=params['batch_size'],
              verbose=1,
              validation_data=[x_val,y_val])
    return history,model

def load_data():
    data_format = '%Y-%m-%dT%H:%M:%SZ'
    CHART_STYLE = 'dark_background'
    START_DATE = datetime.strptime('2019-10-09T00:00:00Z',data_format)
    DAYS = 15

    #repo_url = 'https://github.com/wwu-csci-497/automated-feedback-lab-sample/blob/master/lab3_repo.csv'
    #commit_url = 'https://github.com/wwu-csci-497/automated-feedback-lab-sample/blob/master/lab3_commit.csv'
    repo_url = 'lab3_repo.csv'
    commit_url = 'lab3_commit.csv'

    repo_table = pd.read_csv(repo_url)
    print(f'loaded {len(repo_table)} repositories')
    commit_table = pd.read_csv(commit_url)
    print(f'loaded {len(commit_table)} commits')


    ########################## preprocess data ##############################
    def check_timestamp(s):
        try:
            datetime.strptime(s, data_format)
        except:
            print(f'Timestamp check failed on {s}')
            return False
        return True
    
    def get_day(s):
        date = datetime.strptime(s, data_format)
        days = (date - START_DATE).total_seconds()/(3600*24)
        if days > DAYS:
            raise ValueError()
        return days
    Commit = namedtuple(
            'Commit',
            ['repo_id','day','timestamp','comment','n_additions','n_deletions',
             'test_ratio']
    )
    Repo = namedtuple(
            'Repo',
            ['repo_id','init_day','start_day','test_ratio','struggled_on_day']
    )
    print(f'Preprocessing commits')
    commits = []
    for commit_row in commit_table.itertuples():
        if not check_timestamp(commit_row.timestamp):
            print(f'Commit {commit_row} has None timestamp')
            continue
        try:
            repo_id = commit_row.repo_url_hash
        except:
            repo_id = commit_row.repo_url
    
        try:
            commit = Commit(
                    repo_id=repo_id, day=get_day(commit_row.timestamp),
                    timestamp=commit_row.timestamp,comment=commit_row.comment,
                    n_additions=int(commit_row.n_additions),
                    n_deletions=int(commit_row.n_deletions),
                    test_ratio=(float(commit_row.n_passed)/float(commit_row.n_run)
                        if commit_row.n_run != 0 else 0)
            )
            if not commit.comment == 'Initial commit' and not commit.n_additions == 556:
                commits.append(commit)
        except ValueError:
            continue
    
    print(f'Preprocessing repos')
    repos = []
    for repo_row in repo_table.itertuples():
        timestamp_fp = repo_row.timestamp_fp
        timestamp_sp = repo_row.timestamp_sp
    
        if timestamp_fp == 'None':
            timestamp_fp = '2019-10-9T00:00:00Z'
        if timestamp_sp == 'None':
            timestamp_sp = '2019-10-9T00:00:00Z'
    
        try:
            repo_id = repo_row.repo_url_hash
        except:
            repo_id = repo_row.repo_url
        try:
            struggle_on_day = [
                    getattr(repo_row,f'day_{i+1}') == 1 for i in range(DAYS -1)
            ]
            repos.append(Repo(
                repo_id=repo_id,init_day=get_day(timestamp_fp),
                start_day=get_day(timestamp_sp),
                test_ratio=float(repo_row.percent_passed),
                struggled_on_day=struggle_on_day
            ))
        except ValueError:
            continue
    
    def epoch_to_day(t):
        return (datetime.fromtimestamp(t) - START_DATE).total_seconds() / (3600*24)
    
    
    ############################################ plot stuff ##################################################
    # order by repo name
    repo_name_hash = {}
    for commit in commits:
        if not commit.repo_id in repo_name_hash:
            repo_name_hash[commit.repo_id] = [commit]
        else:
            repo_name_hash[commit.repo_id].append(commit)
    
    repos = []
    x = []
    y = []

    # plot total number of changes vs total number of test casses passed
    for repo in repo_name_hash:
        repos.append(repo)
        commit_lst = repo_name_hash[repo]

        total_changes = 0
        total_additions = 0
        total_deletions = 0

        best_test_ratio = 0
        # build up total_changes and get target best_test_ratio
        for commit in commit_lst:
            total_additions += commit.n_additions
            total_deletions += commit.n_deletions
            total_changes = total_additions + total_deletions
            x.append(np.array([commit.day/DAYS,len(commit.comment),total_changes,total_additions,total_deletions,commit.test_ratio]))
            best_test_ratio = max(best_test_ratio, commit.test_ratio)
        y += ([best_test_ratio] * len(commit_lst))

    return np.array(x),np.array(y)


if __name__=='__main__':
    main()
