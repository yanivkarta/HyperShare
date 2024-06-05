''' import unittest'''
import unittest

import numpy as np
import time

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

#wrapper for the forest:
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
    
    def predict(self, data):
        #predict the anomaly scores for the data
        scores = np.zeros(data.shape[0], dtype=np.float64)
        iso.FastMatrixForest_predict( data.ctypes.data_as(ctypes.c_void_p), scores.ctypes.data_as(ctypes.c_void_p), data.shape[0]) 
        return scores

    def delete(self):
        iso.FastMatrixForest_delete(self.forest)
        return None 
    '''  '''
def create_python_fast_matrix_forest(data, labels = None, n_trees = 1000, max_samples = 1000, max_depth = 1000, n_jobs = 1): 
    #create the forest
    forest = FastMatrixForest(n_trees, max_samples, n_jobs)

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
def test():
    #test the time taken to fit the forest
    start = time.time()
    forest.fit(data, labels)
    end = time.time()
    print('Time taken to fit the forest: ', end-start)

    #test the time taken to predict the anomaly scores
    start = time.time()
    ret = forest.predict(data)
    print('Anomaly scores: ', ret)
    print('Length of the anomaly scores: ', len(ret))
    end = time.time()
    print('Time taken to predict the anomaly scores: ', end-start)

    #test the time taken to delete the forest
    start = time.time()
    forest.delete()
    end = time.time()
    print('Time taken to delete the forest: ', end-start)   
    return None

if __name__ == '__main__':
    test()
    unittest.main()
