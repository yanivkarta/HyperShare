''' import unittest'''
import unittest

import numpy as np
import time
from sklearn.ensemble import IsolationForest

from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split



#import the shared object:
import ctypes
import os
import sys



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
        ret = iso.FastMatrixForest_predict( data.ctypes.data_as(ctypes.c_void_p), scores.ctypes.data_as(ctypes.c_void_p), data.shape[0])         
        return ret
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
    start_time = time.time()
    # Measure time taken to predict anomaly scores
    forest.fit(data, labels)
    end_time = time.time()
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
    print(f"Time taken for the isolation forest to predict anomaly scores: {end_time - start_time} seconds")
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
    
    #print iso vs matrix predictions each line:
    for i in range( scores.shape[0]):
        print(iso_predictions[i], predictions[i],labels[i])

    # Compare predictions to labels
    
    # Print predictions
    
    # Print accuracy
    accuracy = np.mean(predictions == labels)
    iso_accuracy = np.mean(iso_predictions == labels)
    print(f"Accuracy: {accuracy}")
    print(f"Isolation Forest Accuracy: {iso_accuracy}")
    

if __name__ == '__main__':
    for i in range(10):
        test()

    # Run unittest
    
    unittest.main()
