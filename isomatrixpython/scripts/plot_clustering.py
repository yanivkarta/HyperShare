### Plot data information
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import argparse
import json
import seaborn as sns
import sqlite3

from matplotlib import rcParams
from matplotlib import cm
from matplotlib import patches as mpatches  # for custom legends

from imblearn.over_sampling import SMOTE, ADASYN,SMOTEN ,SVMSMOTE 
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours, NeighbourhoodCleaningRule, NearMiss, AllKNN, CondensedNearestNeighbour, OneSidedSelection, RepeatedEditedNearestNeighbours, InstanceHardnessThreshold, ClusterCentroids, NearMiss, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek 
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier

from imblearn import FunctionSampler  # to use a idendity sampler

from collections import Counter




#clustering:
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer


from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification

#vectorizers:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
#bag of words vectorizer
from sklearn.feature_extraction.text import CountVectorizer
#scoring functions:
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

#distance metrics:
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
 
#outlier detection:
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#plotting:
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import SparseCoder
#pipelines
from sklearn.pipeline import Pipeline
#import scipy guassian kde 

import scipy.stats as stats 

vectorizers = [ TfidfVectorizer(), HashingVectorizer(),CountVectorizer() ] 

#clustering algorithms:
clustering_algorithms = [ KMeans(), DBSCAN(), AgglomerativeClustering(), SpectralClustering(), OPTICS(), Birch(), MeanShift(), AffinityPropagation(), MiniBatchKMeans(), FeatureAgglomeration(), AgglomerativeClustering(), OPTICS(), Birch(), MeanShift(), AffinityPropagation(), MiniBatchKMeans() ] 

#dimensionality reduction algorithms:
dimensionality_reduction_algorithms = [ PCA(), TSNE(), MDS(), Isomap(), LocallyLinearEmbedding(), SpectralEmbedding() ]

 
#scalers:
scalers = [ StandardScaler(), MinMaxScaler(), RobustScaler(), Normalizer() ] 
 
#outlier detection algorithms:
outlier_detection_algorithms = [ IsolationForest(n_estimators=500,n_jobs=10), LocalOutlierFactor() ] 

    

def plot_file ( file, normalize=False):
    # Read the data
    df = pd.read_csv(file, sep=',', header=None, index_col=None) 
    #print the dataframe data information:
    print(df.info())
    #print the dataframe data description:
    print(df.describe())

    #assume the last column is the label 
    labels = df.iloc[:,-1]
    
    #remove all the columns with no information 
    df = df.dropna(axis=1, how='all')
    #remove all the rows with no information
    df = df.dropna(axis=0, how='all')
    #remove all the columns with no variance
    df = df.loc[:,df.var() != 0]

    #define X as the data and y as the labels
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]


    #create a plot with each combination of outlier detection algorithm and scaler
    #a vectorizer and a clustering algorithm 
    #default __main__ function opening ./db/provallo_core/provallo_attributes.dat 
    #REMOVE DIMENSIONALITY REDUCTION ALGORITHMS
    for vectorizer in vectorizers:
        for scaler in scalers:
            for outlier_detection_algorithm in outlier_detection_algorithms:
                for clustering_algorithm in clustering_algorithms:
                       
                        #define the number of clusters as the number of unique labels 
                        n_clusters = len(np.unique(y)) 
                        #define the number of dimensions as 2
                        n_dimensions = 2
                        #define the number of outliers as the number of samples - the number of clusters
                        n_outliers = len(X) - n_clusters
                        #define the number of samples
                        n_samples = len(X)
                        #define the number of features
                        n_features = len(X.columns)
                        
                        n_components = n_clusters

                        #create a plot with 4 figures (2x2) 
                        fig,ax= plt.subplots(2,2, figsize=(15,15)) 
                        #use rainbow colors
                        colors = cm.rainbow(np.linspace(0, 1, n_clusters)) 

                        #initialize the clustering algorithm with n_clusters 
                        #if the algorithm is DBSCAN, set the eps to 0.3 and the min_samples to 10 instead of n_clusters 
                        if clustering_algorithm.__class__.__name__ != "DBSCAN": 
                            clustering_algorithm.set_params(n_clusters=n_clusters) 
                        else:
                            clustering_algorithm.set_params(eps=0.3, min_samples=10)
                        

                        #initialize the outlier detection algorithm with n_outliers 
                        #outlier_detection_algorithm.set_params(contamination=(n_outliers/n_samples)) 
                        #set the number of components to n_dimensions 
                        #apply the vectorizer to the data columns that are not the label column 
                        for column in X.columns:
                            #if the column is not already numeric
                            if not (X[column].dtype == np.float64 or X[column].dtype == np.int64):
                                X[column] = X[column].astype(str)
                                #apply the vectorizer to the column
                                X[column] = vectorizer.fit_transform(X[column]) 
                        #make sure the data is numeric 
                        X = X.apply(pd.to_numeric, errors='coerce') 
                        #drop the NaN values
                        #apply the scaler to the data
                        Xscaled = X#scaler.fit_transform(X)
                        #apply the outlier detection algorithm to the data
                        outliers = outlier_detection_algorithm.fit_predict(Xscaled)
                        #apply the clustering algorithm to the data
                        clusters = clustering_algorithm.fit_predict(Xscaled)
                        
                        Xreduced = Xscaled


                        #score can be pca / tsn / mds / isomap / lle / se 
                        score = silhouette_score(Xscaled, clusters, metric='euclidean') 
                        #print the silhouette score
                        print("[+]Silhouette score: ", score)
                        #score can be pca / tsn / mds / isomap / lle / se
                        score = calinski_harabasz_score(Xscaled, clusters)
                        #print the calinski harabasz score
                        print("[+]Calinski harabasz score: ", score)
                        #score can be pca / tsn / mds / isomap / lle / se
                        score = davies_bouldin_score(Xscaled, clusters)
                        #print the davies bouldin score
                        print("[+]Davies bouldin score: ", score)

                        #subplots show performance of the clustering algorithm with the outlier detection algorithm 
                        #and the scaler
                        #create a meshgrid from the data 
                        s = np.linspace(Xscaled.min().min(), Xscaled.max().max(), 100)
                        #create a contour from the data
                        xx, yy = np.meshgrid(s, s)
                        #create a contour from the data
                        contour = np.c_[xx.ravel(), yy.ravel()] 
                        #create a gaussian kde from the data
                        Z = stats.gaussian_kde(contour.T)(contour.T) 
                        #reshape the gaussian kde from the data
                        Z = Z.reshape(xx.shape)
                        #create a contourf from the data
                        ax[0,0].contourf(xx, yy, Z , cmap=cm.Blues_r)
                        ax[0,1].contourf(xx, yy, Z , cmap=cm.Blues_r)
                        ax[1,0].contourf(xx, yy, Z , cmap=cm.Blues_r)
                        ax[1,1].contourf(xx, yy, Z , cmap=cm.Blues_r)
                        
                        #centroids are the mean values of the clustering algorithm predictions 
                        centroids = np.array([Xscaled[clusters == i].mean(axis=0) for i in range(n_clusters)]) 
                        
                        #plot the centroids [centroids] 
                        #overlay with sns scatterplot [sns.scatterplot] and opacity 0.5 [alpha=0.5] 
                        #use the colors defined earlier [colors]
                        #use black edges [edgecolor='k']
                        #use the ax[0,0] subplot
                        g =sns.scatterplot(x=centroids[:,0], y=centroids[:,1], ax=ax[0,0], palette=colors,  alpha=0.5, edgecolor='k') 
                        #remove the legend
                        

                        #plot the scatterplot of the data [Xscaled]
                        #overlay with sns scatterplot [sns.scatterplot] and opacity 0.5 [alpha=0.5]
                        #use the colors defined earlier [colors]
                        #use black edges [edgecolor='k']
                        #use the ax[0,1] subplot
                        g =sns.scatterplot(x=Xscaled.iloc[:,0], y=Xscaled.iloc[:,1], ax=ax[0,1], palette=colors,  alpha=0.5, edgecolor='k')
                        
                        
                        #plot the scatterplot of the data [Xscaled]
                        #overlay with sns scatterplot [sns.scatterplot] and opacity 0.5 [alpha=0.5]
                        #use the colors defined earlier [colors]
                        #use black edges [edgecolor='k']
                        #use the ax[1,0] subplot
                        g = sns.scatterplot(x=Xscaled.iloc[:,0], y=Xscaled.iloc[:,1], ax=ax[1,0], palette=colors,  alpha=0.5, edgecolor='k')
                        
                        
                        #plot the scatterplot of the data [Xscaled]
                        #overlay with sns scatterplot [sns.scatterplot] and opacity 0.5 [alpha=0.5]
                        #use the colors defined earlier [colors]
                        #use black edges [edgecolor='k']
                        #use the ax[1,1] subplot
                        g = sns.scatterplot(x=Xscaled.iloc[:,0], y=Xscaled.iloc[:,1], ax=ax[1,1], palette=colors,  alpha=0.5, edgecolor='k')
                        
                        
                        #iterate over the unique labels [np.unique(y)]
                        
                        #save the plot
                        plt.savefig("./plots/" + vectorizer.__class__.__name__ + "_" + scaler.__class__.__name__ + "_" + outlier_detection_algorithm.__class__.__name__ + "_" + clustering_algorithm.__class__.__name__ +  ".png")
                        #close the plot
                        plt.close()

 
 

                        
    # ===================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data information')
    parser.add_argument('file', type=str, help='file to plot')
    args = parser.parse_args()
    plot_file(args.file)


    