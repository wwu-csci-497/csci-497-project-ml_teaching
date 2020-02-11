from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.losses import mean_squared_error,mean_absolute_error
from keras.activations import relu,elu,linear
from keras.optimizers import Adam,Nadam,RMSprop
from talos.model import lr_normalizer
import numpy as np
import pickle
import deepchem as dc
import talos as ta

#Global constants for qm8
fp_len=1024
#last layer neuron number
label_len=16
#grid_downsample for random search
grid_downsample=0.01

# parameters for talos
p = {'lr': (0.05, 0.5, 5),
     'neuron_tuple':[[50,25,20],[25,20],[20]],
     'batch_size': (2, 10, 30),
     'epochs': [100,200],
     'dropout': (0, 0.5, 5),
     'optimizer': [Adam, Nadam, RMSprop],
     'losses': [mean_squared_error,mean_absolute_error],
     'activation':[relu, elu],
     'last_activation': [linear]}

def main():
    train_set,valid_set,test_set=load_data()
    t=ta.Scan(x=train_set[0],
              y=train_set[1],
              model=fp_model,
              grid_downsample=grid_downsample,
              params=p,
              dataset_name='qm8',
              experiment_no='1')
    pickle.dump(t,open('hyper.pk','wb'))

def load_data():
    #'Raw'=featurize as rdkit objects
    qm8_tasks,qm8_datasets,transformers=dc.molnet.load_qm8(featurizer='ECFP',fp_len=fp_len)
    train_dataset,valid_dataset,test_dataset=qm8_datasets
    #train
    X_train=train_dataset.X
    y_train=train_dataset.y
    #valid
    X_valid=valid_dataset.X
    y_valid=valid_dataset.y
    #test
    X_test=test_dataset.X
    y_test=test_dataset.y
    return (X_train,y_train),(X_valid,y_valid),(X_test,y_test)

def fp_model(x_train,y_train,x_val,y_val,params):
    # To train
    model=Sequential()
    layers=params['neuron_tuple']
    num_h=len(layers)
    for i in range(num_h):
        num_neurons=layers[i]
        model.add(Dense(num_neurons,activation=params['activation']))
        model.add(Dropout(params['dropout']))
    model.add(Dense(label_len,activation=params['last_activation']))
    model.compile(optimizer=params['optimizer'](lr=ta.model.lr_normalizer(params['lr'],params['optimizer'])),
                  loss=params['losses'],
                  metrics=['acc'])

    #Train the model, iterating on the data in batches of 32 samples
    history=model.fit(x_train,y_train,
              epochs=params['epochs'],
              batch_size=params['batch_size'],
              validation_data=[x_val,y_val],
              verbose=1)
    return history,model

if __name__=='__main__':
    main()
