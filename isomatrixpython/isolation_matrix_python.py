
'''
This file contains the python wrapper for the isolation matrix algorithm. The algorithm is implemented in C++ and the python wrapper is created using ctypes. The algorithm is implemented in the file isolation_matrix.cpp and the shared object file is created using the command:
g++ -shared -o libisolation_mat_python.so -fPIC FastMatrixForest.cpp PythonFastMatrixForest.cpp -std=c++11 -O3 -march=native -fopenmp -I/usr/include/python3.6m -lpython3.6m -lboost_python3 -lboost_numpy3 -lboost_system -lboost_filesystem   
The implementation wraps the C++ code in a python class. The class has the following methods: 
1. __init__: This method initializes the parameters of the forest.
2. fit: This method fits the forest to the data.
3. predict: This method predicts the anomaly scores for the data.
4. __del__: This method deletes the forest.
The class is created using the ctypes library. The shared object file is loaded using ctypes.CDLL. The data is passed to the C++ code as a pointer to a contiguous array. The anomaly scores are returned as a numpy array. 

'''

import numpy as np
import ctypes
import os
import sys
import time

#load the .so file
path = os.path.dirname(os.path.abspath(__file__)) 
lib = ctypes.CDLL(path + '/libisolation_mat_python.so') 
#import class from libisolation_mat_python.so (FastMatrixForest_delete, FastMatrixForest_fit, FastMatrixForest_predict , FastMatrixForest_create_python_fast_matrix_forest , FastMatrixForest) 
#class for the  isolation matrix algorithm

class FastMatrixForest(object):
    forest = None
    data = None
    n_trees = None
    max_samples = None
    n_features = 0
    max_depth = 0
    n_jobs = 1 

    #attributes:
    #n_trees: number of trees in the forest
    #max_samples: maximum number of samples in each tree
    #n_features: number of features in the data
    #max_depth: maximum depth of the trees
    #n_jobs: number of threads to use
    #forest: pointer to the forest object
    #data: pointer to the data

    def __init__(self, n_trees=1000,max_depth=1000, n_jobs=1):
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
            self.forest = lib.FastMatrixForest_create_python_fast_matrix_forest(self.data.ctypes.data_as(ctypes.c_void_p),self.labels.ctypes.data_as(ctypes.c_void_p))   
 
        #fit the forest to the data
        self.results = np.zeros(data.shape[0], dtype=np.float64)
        self.n_samples = data.shape[0] 
        self.n_features = data.shape[1]
        lib.FastMatrixForest_fit( self.forest,  data.ctypes.data_as(ctypes.c_void_p), self.results.ctypes.data_as(ctypes.c_void_p), data.shape[0]) 

        return self
    
    def predict(self, data):
        #predict the anomaly scores for the data
        scores = np.zeros(data.shape[0], dtype=np.float64)
        scores = lib.FastMatrixForest_predict( self.forest,  data.ctypes.data_as(ctypes.c_void_p), scores.ctypes.data_as(ctypes.c_void_p), data.shape[0]) 

        
        return scores
    
    def delete(self):
        #delete the forest
        lib.FastMatrixForest_delete(self.forest)
        self.forest = None
        return None
    
def create_python_fast_matrix_forest(data, labels = None, n_trees = 1000, max_samples = 1000, max_depth = 1000, n_jobs = 1): 
    #create the forest
    forest = FastMatrixForest(n_trees, max_samples, n_jobs)

    return forest

if __name__ == '__main__':

    #test the isolation matrix algorithm
    data = np.random.rand(1000, 10)
    forest = create_python_fast_matrix_forest(data,  np.random.randint(0, 2, 1000), 100, 100, 100, 1) 
    forest.fit(data)
    scores = forest.predict(data) 
    print(scores)
    del forest
    print("done")   
