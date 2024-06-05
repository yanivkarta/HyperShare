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

from sklearn.datasets import make_classification

#import scipy guassian kde

import scipy.stats as stats 
# Global variables
# ====================

## read file (db/provallo_core/provallo_attributes.dat) attribute collection.
## display sampling/resampling of samples from .dat files 
## using smote/adasyn/etc.... 


def plot_file ( file, normalize=False):
    # Read the data
    # ===================
    #open file 
    if file.endswith(".db"):
        #sqlite file, read the data using pandas read_sql function 
        conn = sqlite3.connect(file)
        df = pd.read_sql("SELECT * FROM log", conn) 
        #vectorize the data fields 
        df = df.apply(pd.to_numeric, errors='coerce')
        #drop the NaN values
        df = df.dropna()


    else:
        df = pd.read_csv(file, sep=',', header=None, index_col=None) 
    #print the dataframe data information:
    print(df.info()) 
    #print the dataframe data description:
    print(df.describe())
    #print the dataframe data head:
    print(df.head())
    #print the dataframe data tail:
    print(df.tail())
    #print the dataframe data shape:
    print(df.shape)
    #print the dataframe data columns:
    print(df.columns)
    #print the dataframe data index:
    print(df.index)
    
    #define 15 samplers
    samplers = [
        SMOTE(random_state=666),
        SVMSMOTE(random_state=666),
        ADASYN(random_state=666),
        SMOTEN(random_state=666),
        RandomUnderSampler(random_state=666),
        TomekLinks(),
        EditedNearestNeighbours(),
        NeighbourhoodCleaningRule(),
        NearMiss(),
        AllKNN(),
        CondensedNearestNeighbour(),
        OneSidedSelection(),
        RepeatedEditedNearestNeighbours(),
        InstanceHardnessThreshold(),
        BalancedBaggingClassifier(random_state=666),
        BalancedRandomForestClassifier(random_state=666),
        EasyEnsembleClassifier(random_state=666),
        RUSBoostClassifier(random_state=666),
        ClusterCentroids(random_state=666),
    
    

    ]
    
    #create a plot with 16 figures (4x4) 
    fig, ax = plt.subplots(4,4, figsize=(15,15)) 
    #set the class labels as the last column 
    #map the colors to the  5 class labels 
    colors = {0:'red', 1:'blue', 2:'green', 3:'yellow', 4:'orange', 5:'purple'} 

    #drop columns where len of unique values is 1 
    df = df.drop(df.columns[df.apply(lambda col: col.nunique() == 1)], axis=1)

    #set the class labels as the last column 
    labels = df.columns[-1]
    n_classes = len(df[labels].unique()) 
    print("[+]Loaded dataset with n_classes: ", n_classes)
    #set the class labels as the last column
    classes = df[labels]
    
    X = df.drop(labels,axis=1)

    y = labels

    #create a linear space from the data
    s = np.linspace(X.min().min(), X.max().max(), 100)
    #create meshgrid from the data
    xx, yy = np.meshgrid(s, s)
    #create contour from the data
    contour = np.c_[xx.ravel(), yy.ravel()]
    #create gaussian kde from the data
    Z = stats.gaussian_kde(contour.T)(contour.T)
       #reshape the gaussian kde from the data
    Z = Z.reshape(xx.shape)
    #create contourf from the data
    ax[0,0].contourf(xx, yy, Z , cmap=cm.Reds_r)

    #create tsne plot from the data
    

    #contour,linear space and meshgrid from the data:
    xx, yy = np.meshgrid(s, s)

    #iterate over the 16 figures
    for i in range(4):
        for j in range(4):
            
            if i+j==0:
                #prepare X for SVD
                X = df.drop(labels,axis=1)
                X = X.values
                #print the shape of X
                print("[+]X shape: ", X.shape)

                #plot scatter plot of the original data over the gaussian kde contour 
                #use SVD to reduce the dimensionality of the data to 2D 
                print("[+]Plotting original data")
                S,V,D = np.linalg.svd(X, full_matrices=False , compute_uv=True)) 
                print("[+]SVD shape: ", S.shape)
                print("[+]SVD shape: ", V.shape)
                print("[+]SVD shape: ", D.shape)
                
                #plot the scatter plot of the original data over the gaussian kde contour
                

                for label in colors:
                    #set X and y to the original data
                    #plot the scatter plot of the SVD reduced data over the gaussian kde contour 
                    g = sns.scatterplot(x=S[:,0], y=S[:,1], hue=classes, ax=ax[i,j], palette=colors,  alpha=0.5, edgecolor='k') 
                    g.get_legend().remove()

                    #set the title of the plot
                    ax[i,j].set_title("Original data")
                    
            else:
                try:
                    #plot contour of the original data over the samplers results 
                    index=i*4+j-1

                    sampler = samplers[index] 

                    print(" plotting sampler: ", sampler.__class__.__name__ )
                    

                    #fit the sampler to the data
                    X_resampled, y_resampled = sampler.fit_resample(df.iloc[:,:-1], classes) 

                    #create contourf plot of the gaussian kde of the original data
                    cmap ={0:'red', 1:'blue', 2:'green', 3:'yellow', 4:'orange', 5:'purple', 6:'olive' , 7:'pink', 8:'brown', 9:'gray' } 
                    cs = [cmap[i] for i in y_resampled] 
                    
                    
                    S,V,D = np.linalg.svd(X_resampled, full_matrices=False, compute_uv=True) 

                    #plot the contour of the original data over the samplers results
                    for label in colors:
                        #set X and y to the resampled data 
                        
                        y = y_resampled
                        ax[i,j].legend([],[], frameon=False)
                        g = sns.scatterplot(x=S[:,0], y=S[:,1], hue=y, ax=ax[i,j], palette=colors,  alpha=0.5, edgecolor='k')
                        g.get_legend().remove()

                    

                    #set the title of the plot
                    ax[i,j].set_title(sampler.__class__.__name__)
                
                
                except IndexError:
                    pass
                except ValueError:
                    pass
                except AttributeError:
                    pass
                except TypeError:
                    pass

                    
 
    
    #save the figure to file
    plt.savefig(file + ".png", dpi=300)
    #close the figure
    plt.close()
    #plt.show()
    # ===================

    #default __main__ function opening ./db/provallo_core/provallo_attributes.dat 
    # ===================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data information')
    parser.add_argument('file', type=str, help='file to plot')
    args = parser.parse_args()
    plot_file(args.file)


    