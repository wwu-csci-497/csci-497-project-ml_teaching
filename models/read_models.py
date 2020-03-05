import csv
import glob
import pandas as pd


def main():
    csv_files = glob.glob('comp_edu2/*')
    csv_files.sort()
    for csv_file in csv_files:
        print('Loading...',csv_file)
        df = pd.read_csv(csv_file,sep=',')
        #max_acc = df["val_accuracy"]
        #print(df.columns)
        #for col in df.columns:
        #    print (col)





if __name__=='__main__':
    main()
