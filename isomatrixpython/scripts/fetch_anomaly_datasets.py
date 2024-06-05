
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_covtype, fetch_kddcup99, fetch_openml
from sklearn.preprocessing import LabelBinarizer 
from sklearn.utils import shuffle as sh 
from sklearn.model_selection import train_test_split 

datasets = [ "http", "smtp", "SA", "SF", "shuttle", "forestcover"] 
''' save the datasets locally to evaluate with hyperpane matrix and with isolation forest ''' 
def print_outlier_ratio(y):
    """
    Helper function to show the distinct value count of element in the target.
    Useful indicator for the datasets used in bench_isolation_forest.py.
    """
    uniq, cnt = np.unique(y, return_counts=True)
    print("----- Target count values: ")
    for u, c in zip(uniq, cnt):
        print("------ %s -> %d occurrences" % (str(u), c))
    print("----- Outlier ratio: %.5f" % (np.min(cnt) / len(y)))



def fetch_anomaly_datasets():
    for dat in datasets:
        print("fetching ", dat) 
        if dat == "forestcover":
            covtype = fetch_covtype()
            X = covtype.data
            y = covtype.target
            s = (y==2)+(y==4)
            X=X[s,:]
            y=y[s]
            y=(y!=2).astype(np.int64)
            print_outlier_ratio(y)

        elif dat == "shuttle":
            shuttle = fetch_openml("shuttle", cache=True ,as_frame=False )
            X = shuttle.data
            y = shuttle.target.astype(np.int64) 
            #X,y=sh(X, y, random_state=random_state)
            #dont shuffle shuttle
            s = (y!=4)
            X=X[s,:]
            y=y[s]
            y=(y!=1).astype(np.int64)
            print_outlier_ratio(y)


        elif dat in["http", "smtp", "SA", "SF"]: 
            kddcup99 = fetch_kddcup99(subset=dat, percent10=False, random_state=42, shuffle=True)
            X = kddcup99.data
            y = kddcup99.target 
        else:
            X, y = fetch_openml(dat,  cache=True, return_X_y=True)    
        
        print("--- Vectorizing data...")
        if dat == "SF":
            lb = LabelBinarizer()
            x1 = lb.fit_transform(X[:, 1].astype(str))
            X = np.c_[X[:, :1], x1, X[:, 2:]]
            y = (y != b"normal.").astype(int)
            print_outlier_ratio(y)


        if dat == "SA":
            lb = LabelBinarizer()
            x1 = lb.fit_transform(X[:, 1].astype(str))
            x2 = lb.fit_transform(X[:, 2].astype(str))
            x3 = lb.fit_transform(X[:, 3].astype(str))
            X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
            y = (y != b"normal.").astype(int)
            print_outlier_ratio(y)

        if dat in ("http", "smtp"):
            y = (y != b"normal.").astype(int)
            print_outlier_ratio(y)

        print("saving ", dat) 
        '''save x/y as comma separated values instead of numpy arrays''' 
        #prepare X for saving as csv

        
        np.savetxt("./db/anomaly/" + dat + "_X.csv", X, delimiter=",") 
       
        #prepare y for saving as csv 
        y = y.reshape(-1,1) 

        np.savetxt("./db/anomaly/" + dat + "_y.csv", y, delimiter=",") 
        

        
        print("done")
    return
''' main '''

if __name__ == "__main__":
    fetch_anomaly_datasets()
