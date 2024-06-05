''' load the pre-fetched datasets saved on fetch_anomaly_datasets.py ''' 
''' compare the scoring functions of isoforest to the saved scoring functions of the hyperplae classifier'''
''' compare the accuracy of the hyperplane classifier to the accuracy of the isolation forest classifier''' 
from time import time
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.ensemble import IsolationForest 
from sklearn.utils import shuffle as sh 
#metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc 
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import average_precision_score 
from sklearn.metrics import classification_report 

from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import PrecisionRecallDisplay 
from sklearn.metrics import RocCurveDisplay 
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay 
from sklearn.inspection import DecisionBoundaryDisplay
#sns
import seaborn as sns
from matplotlib import cm

#color cycle : 
from itertools import cycle 
#save and load numpy arrays

from numpy import savetxt 
from numpy import loadtxt


#datasets

n_jobs=40
datasets = [ "http", "smtp", "SA", "SF", "shuttle", "forestcover"] 
prefix = "./db/anomaly/"


def print_iso_forest_over_datasets():
    #same as the comparison, just load the isolation forest and print the accuracy of the isolation forest without the plotting 
    for dat in datasets:
        #load the X and y from the files 
        print("loading file ",prefix+ dat + "_X.csv")
        X = np.loadtxt(prefix+ dat + "_X.csv", delimiter=",")
        print(X.shape)
        y = np.loadtxt(prefix+ dat + "_y.csv", delimiter=",")
        print("loaded ",prefix+ dat + "_X.csv")
        print("loaded ",prefix+ dat + "_y.csv")
        print("loaded ",X.shape, y.shape)
        print("----- Target count values: ")
        uniq, cnt = np.unique(y, return_counts=True)
        for u, c in zip(uniq, cnt):
            print("------ %s -> %d occurrences" % (str(u), c))
        print("----- Outlier ratio: %.5f" % (np.min(cnt) / len(y)))
        n_samples, n_features = X.shape
        n_outliers = np.sum(y == 1)
        n_inliers = np.sum(y == 0)

        print("n_samples %d, \nn_features %d, \nn_outliers %d, \nn_inliers %d" % (n_samples, n_features, n_outliers, n_inliers)) 
        print("X.shape", X.shape)
        print("y.shape", y.shape)
        print("X", X)
        print("y", y)
        X_train = X[:n_samples // 2]
        X_test = X[n_samples // 2:]
        y_train = y[:n_samples // 2]
        y_test = y[n_samples // 2:]
        print("fitting isolation forest")
        t0 = time()
        clf = IsolationForest(random_state=42, n_jobs=n_jobs, verbose=1, contamination=0.1, n_estimators=500, max_samples=256 ) 
        clf.fit(X_train, y_train) 
        scoring = -clf.decision_function(X_test) 
        t1 = time()
        #print fit accuracy
        print ("fitting isolation forest took ", t1-t0)
        
      
        print("predicting isolation forest")
        t0 = time()
        y_pred = clf.predict(X_test)
        t1 = time()
        print("predicting isolation forest took ", t1-t0)
      
        #acalculate the accuracy of the fitness based on y_train
        accuracy = y_pred[y_pred == y_test].shape[0] / y_test.shape[0] 
        accuracy = 1 - accuracy

        print("accuracy: ", accuracy)
        
        print("accuracy: ", accuracy)
        auc = roc_auc_score(y_test, scoring) 
        print("auc: ", auc) 
        #print("accuracy: ", clf.score(X_train, y_train)) 



          #print confusion matrix of the predictions
        print("confusion matrix: ", confusion_matrix(y_test, y_pred))
        #evaluate the predictions



#%datasets already fetched and saved under ./db/anomaly/ as _X.npy and _y.npy files and as _X.csv and _y.csv files 
def compare_hyperiso():
    #print_iso_forest_over_datasets()
    for dat in datasets:
        print("loading file ",prefix+ dat + "_X.csv")
        X = np.loadtxt(prefix+ dat + "_X.csv", delimiter=",") 
        print(X.shape)
        y = np.loadtxt(prefix+ dat + "_y.csv", delimiter=",") 
        print("loaded ",prefix+ dat + "_X.csv") 
        print("loaded ",prefix+ dat + "_y.csv") 
        print("loaded ",X.shape, y.shape)
        getchar = input("press any key to continue") 

        print("----- Target count values: ") 
        uniq, cnt = np.unique(y, return_counts=True) 
        for u, c in zip(uniq, cnt):
            print("------ %s -> %d occurrences" % (str(u), c))
        print("----- Outlier ratio: %.5f" % (np.min(cnt) / len(y)))
        n_samples, n_features = X.shape 
        n_outliers = np.sum(y == 1)
        n_inliers = np.sum(y == 0)
        print("n_samples %d, \nn_features %d, \nn_outliers %d, \nn_inliers %d" % (n_samples, n_features, n_outliers, n_inliers)) 
        print("X.shape", X.shape)
        print("y.shape", y.shape)

        score_prefix = "./df/"
        prediction_prefix = "./pred/"
        #load score and y_pred of hyperplane classifier from files on the paths below:
        print("loading hyperplane classifier score and hyperplane y_pred from files") 
        hyperplane_score = np.loadtxt(score_prefix+ dat + "_decision_function.csv", delimiter=",") 
        hyperplane_y_pred = np.loadtxt(prediction_prefix + dat + "_predictions.csv", delimiter=",") 
        print("loaded hyperplane classifier score and hyperplane y_pred from files") 
        print("hyperplane_score.shape", hyperplane_score.shape) 

        print("hyperplane_y_pred.shape", hyperplane_y_pred.shape)

        #normalize -inf
        #-1 are anomalies, 1 are normal 
        hyperplane_y_pred[hyperplane_y_pred == -np.inf] = -1 
        hyperplane_y_pred[hyperplane_y_pred ==np.inf] = 1 
        hyperplane_y_pred[hyperplane_y_pred == 0] = 1 
        hyperplane_y_pred[hyperplane_y_pred == 1] = 0 

        #print("hyperplane_score", hyperplane_score) 
        
        #fit and predict the isolation forest 
        print("fitting isolation forest")
        t0 = time()
        clf = IsolationForest(random_state=42, n_jobs=n_jobs, verbose=1, contamination=0.1, n_estimators=500, max_samples=256 )
        clf.fit(X[n_samples//2:], y[n_samples // 2:])

        scoring_iso = clf.decision_function(X)
        t1 = time()
        #print fit accuracy
        print ("fitting isolation forest took ", t1-t0) 
        print("predicting isolation forest") 
        t0 = time() 
        y_pred = clf.predict(X)
        t1 = time()
        print("predicting isolation forest took ", t1-t0)
        #acalculate the accuracy of the fitness based on y_train
        iso_accuracy = y_pred[y_pred == y].shape[0] / y.shape[0] 
        iso_accuracy = 1 - iso_accuracy
        #print the shape of the iso score
        print("scoring_iso.shape", scoring_iso.shape) 

        print("iso_accuracy: ", iso_accuracy)

        hyperplane_accuracy = hyperplane_y_pred[hyperplane_y_pred == y].shape[0] / y.shape[0] 
        #hyperplane_accuracy = 1 - hyperplane_accuracy 
         
        print("hyperplane_accuracy: ", hyperplane_accuracy) 
        #print("accuracy: ", clf.score(X_train, y_train))
        #
        #print confusion matrix of the predictions 

        #draw auc_roc curve and save it to file
        iso_fpr, iso_tpr, _ = roc_curve(y, -scoring_iso) 
        auc_iso = auc(iso_fpr, iso_tpr) 
        print("auc_iso: ", auc_iso) 
        
        #draw auc_roc curve and save it to file
        hyperplane_score[hyperplane_score==np.inf] = 0  
        hyperplane_score[hyperplane_score==-np.inf] = 1
        hyperplane_score[hyperplane_score==-1]  = 1 

        hyper_fpr, hyper_tpr, _ = roc_curve(y, hyperplane_score) 

        print("hyper_fpr: ", hyper_fpr.shape) 
        print("hyper_tpr: ", hyper_tpr.shape)
         

        #normalize tpr/fpr

        auc_hyperplane = auc(hyper_fpr, hyper_tpr) 

        print("auc_hyperplane: ", auc_hyperplane) 


        #sns.drawline()
        lw = 2
        plt.figure(figsize=(10,10)) 
        
        plt.plot(iso_fpr, iso_tpr, color='darkred',
                    lw=lw, label='Isolation Forest ROC curve (area = %0.2f),(accuracy= %0.2f)' % (auc_iso, iso_accuracy ))
        plt.plot(hyper_fpr, hyper_tpr, color='darkblue',
                    lw=lw, label='Hyperplane ROC curve (area = %0.2f),(accuracy = %0.2f)' % (auc_hyperplane, hyperplane_accuracy ))
        
                
       # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.grid(True)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate') 
        #add labels for each legend
        plt.legend(loc="lower right") 
        plt.title('Receiver operating characteristic '+dat) 
        plt.savefig( "iso_hyper_comparison"+dat+".png")  

 

if __name__ == '__main__':
    compare_hyperiso()

