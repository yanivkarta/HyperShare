''' import unittest'''
import unittest

import numpy as np
import time
from sklearn.ensemble import IsolationForest

from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split


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
#sns 
import seaborn as sns

#matplotlib
import matplotlib.pyplot as plt

#png    
import matplotlib.image as mpimg

#import the shared object:
import ctypes
import os
import sys

#dataframe
import pandas as pd
#make_classification: 
from sklearn.datasets import make_classification, make_blobs, make_moons 



path = os.path.dirname(os.path.abspath(__file__))

iso = ctypes.CDLL('./libisolation_mat_python.so')

if iso == None:
    print('Could not load the shared object')
    sys.exit()
else:
    print('Shared object loaded successfully')

#try to import the class
#import isolation_matrix_python as iso

#main test
#enum the methods,attributes,functions signatures of isolation_matrix_python 

print ('Methods of isolation_matrix_python: ', [f for f in dir(iso) if callable(getattr(iso, f))]) 
print('Attributes of isolation_matrix_python: ', [f for f in dir(iso) if not callable(getattr(iso, f))]) 
print('-----------------------------')

#fit the forest
data = np.random.rand(1000, 100) 
labels = np.random.randint(0, 2, 1000)
mat_results = np.zeros(1000)
iso_results = np.zeros(1000)
#wrapper for the forest:
class FastMatrixForest(object):
    forest = None
    data = None
    n_trees = None
    max_samples = None
    n_features = 0
    max_depth = 0
    n_jobs = 1 
    labels = None
    #attributes:
    #n_trees: number of trees in the forest
    #max_samples: maximum number of samples in each tree
    #n_features: number of features in the data
    #max_depth: maximum depth of the trees
    #n_jobs: number of threads to use
    #forest: pointer to the forest object
    #data: pointer to the data

    def __init__(self,data=None, labels = None, n_trees=1000,max_depth=1000, n_jobs=1):
        self.n_trees = n_trees
        self.n_features = 1
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.forest = None
        self.data = None
        self.results = None
        self.n_samples = 0
        
    def fit(self, data, labels ):
        self.data = data
        self.labels = labels if labels is not None else np.zeros(data.shape[0], dtype=np.float64) 
        #create the forest
        if self.forest is None:
            #pass the data and labels to the shared object, encode them as np.float64 arrays 
            data_flat = data.flatten().astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)) 
            labels_flat = self.labels.flatten().astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)) 
            self.forest = iso.FastMatrixForest_create_python_fast_matrix_forest(data_flat, labels_flat) 
            #self.forest = iso.FastMatrixForest_create_python_fast_matrix_forest(self.data.ctypes.data_as(ctypes.c_void_p),self.labels.ctypes.data_as(ctypes.c_void_p)) 
             #self.forest = iso.FastMatrixForest_create_python_fast_matrix_forest(self.data ,self.labels.ctypes.data_as(ctypes.c_void_p))   

        #fit the forest to the data
        self.results = np.zeros(data.shape[0], dtype=np.float64)
        self.n_samples = data.shape[0] 
        self.n_features = data.shape[1]

        return self
    
    def predict_proba(self, data):
        #predict the anomaly scores for the data
        scores = np.zeros(data.shape[0], dtype=np.float64)
        ret = iso.FastMatrixForest_get_scores( data.ctypes.data_as(ctypes.c_void_p), scores.ctypes.data_as(ctypes.c_void_p), data.shape[0]) 
        print ("prediction shape: ", scores.shape)
        #print ("ret shape: ", ret)
        
        return scores
    
    def predict(self, data):
        #predict the anomaly scores for the data
        scores = np.zeros(data.shape[0], dtype=np.float64)
        ret =iso.FastMatrixForest_predict( data.ctypes.data_as(ctypes.c_void_p), scores.ctypes.data_as(ctypes.c_void_p), data.shape[0]) 
        print ("prediction shape: ", scores.shape)
        print ("ret shape: ", ret)
        if scores.shape[0] != data.shape[0]:
            print ("error in prediction")
            return ret
        
        for i in range(scores.shape[0]):
            if scores[i] > 0.5:
                scores[i] = 1
            else:
                scores[i] = 0
        return scores
        

    
def create_python_fast_matrix_forest(data, labels = None, n_trees = 1000, max_samples = 1000, max_depth = 1000, n_jobs = 1): 
    #create the forest
    forest = FastMatrixForest( data, labels, n_trees=1000,max_depth=1000, n_jobs=1)
    return forest 




try:
    forest = create_python_fast_matrix_forest(100, 256, 100, 30, 1)
    print('Forest created successfully')
    print(forest)
    #print the attributes of the forest :
    #get the properties of the forest :
    properties = forest.__dict__
    print('Properties of the forest: ', properties)
    #print the function names of the forest: 
    functions = [f for f in dir(forest) if callable(getattr(forest, f))] 
    print('Functions of the forest: ', functions) 
    print('-----------------------------')
    #print the signature of the fit function:
    
    print('Signature of the fit function: ', forest.fit.__annotations__, forest.fit.__doc__, forest.fit.__name__ , forest.fit.__qualname__ )
    #print the signature of the predict function:
    print('Signature of the predict function: ', forest.predict.__annotations__)
    
    print('Fitting the forest')
    #results = forest.fit(data, labels)
    
    try:
        res = forest.fit(data, labels)
        print('Forest fitted successfully')
        

    except Exception as e:
        print('Could not fit the forest: ', e)
        sys.exit() 


    print('Forest fitted successfully')

except Exception as e:
    print('Could not create the forest: ', e)
    sys.exit() 

#main test:
__annotations__ = {'test': 'None'}


def generate(data, labels):
    # Generate test data
    num_samples = 10000
    num_features = 10
    data = np.random.rand(num_samples, num_features)
    labels = np.zeros(num_samples, dtype=np.int8)

    for i in range(num_samples):
        rowsum = 0
        for j in range(num_features):
            rowsum += data[i, j]

        if rowsum//num_features > 0.5:
            labels[i] = 1
        elif rowsum//num_features < 0.0:
            labels[i] = -1
        else:
            labels[i] = 0

    return data, labels
    
    pass


def test():
    """
    Set up test data, print test data, compare to sklearn isolation forest, measure time taken to fit the forest,
    measure time taken to predict anomaly scores, regenerate data and labels, measure time taken to predict anomaly scores,
    translate scores to predictions and compare to the labels, measure time taken to predict anomaly scores, print
    iso vs matrix predictions each line, compare predictions to labels, print predictions, print accuracy, and print
    isolation forest accuracy.

    This function does not take any parameters.

    This function does not return any values.
    """
    # Set up test data
    iso_times = []
    matrix_times = []
    num_samples = 10000
    num_features = 10
    global data, labels
    data = np.random.rand(num_samples, num_features)
    labels = np.zeros(num_samples, dtype=np.int8)
    generate(data, labels)        

    # Print test data
    print("=*" * 20)
    for i in range(10):
        for j in range(10):
            print(data[i, j], end=" ")
        print(" ", end=" ")
        print(labels[i])
    print("=*" * 20)
    #compare to sklearn isolation forest
    print ("Labels sum over samples: ", np.sum(labels)/num_samples) 
    
    iso_forest = IsolationForest(n_estimators=1000, max_samples=num_samples,  n_jobs=1) 

    # Measure time taken to fit the forest
    start_time = time.time()
    iso_forest.fit(data, labels)
    end_time = time.time()

    print(f"Time taken to fit the forest: {end_time - start_time} seconds") 

    #add to the times list
    iso_times.append(end_time - start_time) 

    start_time = time.time()
    # Measure time taken to predict anomaly scores
    forest.fit(data, labels)
    end_time = time.time()

    #add to the times list
    matrix_times.append(end_time - start_time)
    print(f"Time taken to fit the matrix: {end_time - start_time} seconds")

    #regenerate data and labels
    data = np.random.rand(num_samples, num_features)
    generate(data, labels)

    # Measure time taken to predict anomaly scores
    #print labels sum over samples again:   
    print ("Labels sum over samples: ", np.sum(labels)/num_samples)
    start_time = time.time()
    scores = iso_forest.predict(data) 
    end_time = time.time()
    #print time taken to predict anomaly scores
    print(f"Time taken for the isolation forest to predict anomaly scores: {end_time - start_time} seconds")
    #add to the times list
    iso_times.append(end_time - start_time)
    #translate scores to predictions and compare to to the labels
    iso_predictions = []
    for i in range(scores.shape[0]):
        if scores[i] > 0.5:
            iso_predictions.append(1)
        else:
            iso_predictions.append(0)

    # Measure time taken to predict anomaly scores

    start_time = time.time()
    predictions = forest.predict(data)
    end_time = time.time()
    print(f"Time taken for the matrix to predict anomaly scores: {end_time - start_time} seconds")
    #add to the times list
    matrix_times.append(end_time - start_time)
    #print iso vs matrix predictions each line:
    for i in range( scores.shape[0]):
        print(iso_predictions[i], predictions[i],labels[i])



    #compare predictions to labels for isolation forest vs matrix
    iso_correct_predictions = 0
    matrix_correct_predictions = 0

    for i in range(scores.shape[0]):
        if iso_predictions[i] == labels[i]:
            iso_correct_predictions += 1    
        if predictions[i] == labels[i]:
            matrix_correct_predictions += 1

    iso_accuracy = iso_correct_predictions/scores.shape[0]
    matrix_accuracy = matrix_correct_predictions/scores.shape[0]
    print("Accuracy of isolation forest: ", iso_accuracy)
    print("Accuracy of matrix: ", matrix_accuracy)

    #print confusion matrix for iso vs matrix
    print("Confusion matrix of isolation forest: ",confusion_matrix(iso_predictions, labels))
    print("Confusion matrix of matrix: ", confusion_matrix(predictions, labels)) 
    return (iso_accuracy, matrix_accuracy, labels, iso_predictions, predictions, iso_times, matrix_times)  



    

    
if __name__ == '__main__':
    
    results = dict()
    for i in range(10):
        results[i] = test()

    #draw results for all 10 tests in one graph
    # set up lines for the accuracy of iso vs matrix
    iso_accuracy = [results[0][0], results[1][0], results[2][0], results[3][0], results[4][0], results[5][0], results[6][0], results[7][0], results[8][0], results[9][0]] 
    matrix_accuracy = [results[0][1], results[1][1], results[2][1], results[3][1], results[4][1], results[5][1], results[6][1], results[7][1], results[8][1], results[9][1]] 
    labels = [results[0][2], results[1][2], results[2][2], results[3][2], results[4][2], results[5][2], results[6][2], results[7][2], results[8][2], results[9][2]] 
    iso_predictions = [results[0][3], results[1][3], results[2][3], results[3][3], results[4][3], results[5][3], results[6][3], results[7][3], results[8][3], results[9][3]] 
    predictions = [results[0][4], results[1][4], results[2][4], results[3][4], results[4][4], results[5][4], results[6][4], results[7][4], results[8][4], results[9][4]] 
    iso_times = [results[0][5], results[1][5], results[2][5], results[3][5], results[4][5], results[5][5], results[6][5], results[7][5], results[8][5], results[9][5]]
    matrix_times = [results[0][6], results[1][6], results[2][6], results[3][6], results[4][6], results[5][6], results[6][6], results[7][6], results[8][6], results[9][6]] 

    #set up the graph
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    fig, ax = plt.subplots()
    ax2 = ax.twinx() 

    #distribute ax/ax2 in the same space over figure
    fig.subplots_adjust(right=0.75) 

    #add subplot for accuracy

    ax.plot(x, iso_accuracy, '-bo', label='IsoForest')
    ax.plot(x, matrix_accuracy, '-ro', label='FastMatrixForest')
    #add subplot for times
    #ax2 = ax.twinx()
    
    ax2.plot(x, iso_times, '-go', label='IsoForest Times')
    ax2.plot(x, matrix_times, '-yo', label='FastMatrixForest Times')  
    
    ax.set_xlabel('Number of Tests')
    ax.set_ylabel('Accuracy')
    ax2.set_ylabel('Time Taken ')
    
    #ax.set_title('Accuracy of IsoForest vs FastMatrixForest')
    #ax2.set_title('Time Taken of IsoForest vs FastMatrixForest' )

    ax.set_xticks(x)
    ax.set_xticklabels(x)

    ax2.set_xticks(x)
    ax2.set_xticklabels(x)

    max_time = max(np.max(iso_times), np.max(matrix_times)) 

    ax.set_ylim(-0.0000001, 1.0000001)
    ax2.set_ylim(-0, max_time)

    ax2.legend()

    #add legend 
    #draw the times


    ax.legend()
    plt.savefig('AccuracyPerformance.png')
    plt.show()
    plt.close()

    
    #test the accuracy of iso vs matrix over an anomaly detection problem 
    #create classification problem:

    classification_set = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, class_sep=1, random_state=42) 

    #add anomalous data
    classification_set[0][8] = 100
    classification_set[0][9] = 100

    truth = classification_set[1]
    isoforest  = IsolationForest(n_estimators=10, max_samples='auto', contamination=0.1, max_features=1.0) 
    matrix = create_python_fast_matrix_forest(100, 256, 100, 30, 1)

    #first iso forest
    isoforest.fit(classification_set[0], classification_set[1])
    iso_pred = isoforest.predict(classification_set[0])

    #transform probabilities to labels
    iso_pred = np.where(iso_pred > 0, 1, 0)
    

    #second matrix forest
    matrix.fit(classification_set[0], classification_set[1])
    matrix_pred = matrix.predict(classification_set[0])
    matrix_probs = matrix.predict_proba(classification_set[0])
    #calculate accuracy
    iso_accuracy = accuracy_score(truth, iso_pred)
    matrix_accuracy = accuracy_score(truth, matrix_pred)

    print(iso_accuracy)
    print(matrix_accuracy)

    #print confusion matrix
    iso_cm = confusion_matrix(truth, iso_pred)
    matrix_cm = confusion_matrix(truth, matrix_pred)

    #draw auc curve and confusion matrix in 2 subplots over a new figure

    #plot confusion matrix and auc_roc for iso forest :
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax1,ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    #align the subplots 

    
    tpr, fpr, _ = roc_curve(truth, iso_pred) 
    roc_auc = auc(fpr, tpr)
    #set ax left half of the image: 
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0.5, 1)

    ax.plot(fpr, tpr, '-bo', label='ROC curve (area = %0.2f)' % roc_auc)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve of IsoForest vs FastMatrixForest')

    #plot confusion matrix and auc_roc for matrix forest :
    
    tpr, fpr, _ = roc_curve(truth, matrix_pred) 
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, '-ro', label='ROC curve (area = %0.2f)' % roc_auc)
    
    ax.legend()



    #position ax1 in the top right and ax2 in the bottom right and ax on the left
    

    
    #plot confusion matrix for iso forest on ax1 
    ax1.matshow(iso_cm)
    ax1.set_title('IsoForest Confusion Matrix')

    #plot confusion matrix for matrix forest on ax2
    # ax2.matshow(matrix_cm)
    ax2.matshow(matrix_cm)
    ax2.set_title('FastMatrixForest Confusion Matrix')

    #plot confusion matrix for iso forest 
    fig.tight_layout()
    plt.savefig('ConfusionMatrix.png')
    plt.show()
    plt.close()

    unittest.main()
