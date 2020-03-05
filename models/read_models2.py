import talos as ta
import pickle
import tensorflow


def main():

    # For each split
    NUM_SPLIT = 15
    for i in range(1,NUM_SPLIT):
        model_name = 'dnn_'+str(i)+'_'+str(NUM_SPLIT)+'.pk'
        print('Loading...',model_name)
        model = pickle.load(open(model_name,'rb'))




if __name__=='__main__':
    main()
