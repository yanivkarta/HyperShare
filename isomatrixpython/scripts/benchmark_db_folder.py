'''
@author:
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cm as cm
from itertools import cycle

#import classifiers

import pandas as pd
import seaborn as sns
import itertools
import re
import argparse
import yaml
import json
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
from collections import namedtuple

# Global variables
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import naive_bayes

from sklearn import linear_model

from sklearn import pipeline
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.manifold import TSNE
#xgboost
from xgboost import XGBClassifier
#catboost
from catboost import CatBoostClassifier


#gaussian naive bayes
from sklearn.naive_bayes import GaussianNB

#lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#qda
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#knn
from sklearn.neighbors import KNeighborsClassifier
#svc
from sklearn.svm import SVC
#logistic regression
from sklearn.linear_model import LogisticRegression
#mlp
from sklearn.neural_network import MLPClassifier


 


#parse .names file for pandas dataframe

# Global variables

# Functions
def parse_names_file(names_file , folder="./"):
    '''
    Parse the .names file and return a pandas dataframe
    '''
    # Read the .names file
    # ===================
    datastem = names_file.split('.')[0]
    datafile = folder + datastem + '.data'
    testfile = folder + datastem + '.test'


    with open( folder+names_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 

    # Parse the .names file
    # ===================
    # Get the column names
    column_names = []
    data_types = []
    target_values   = None
    target_column = None
    #map of column names to discrete values:
    discrete_column_values   = None

    ''' look for target: '''
    for line in content:
        if line.startswith('|') or line.startswith('#') or line.startswith('!'):
            continue
        elif line.find('target:') != -1:
            target_column = line.split(':')[1].strip()
            ''' remove trailing ':' '''
            
        elif line.find(':') != -1:
            column_name = line.split(':')[0].strip()
            if(target_column is not None and column_name == target_column):
                target_values = line.split(':')[1].strip().split(',')
                target_values = [x.strip() for x in target_values]
                target_values = list(filter(None, target_values))
                column_names.append(line.split(':')[0].strip())
                continue     
            column_names.append(line.split(':')[0].strip())
            if(line.split(':')[1].strip() == 'continuous' or line.split(':')[1].strip() == 'numeric'):
                data_types.append('continuous')
            else:
                data_types.append('discrete')
                if line.split(':')[1].find(',') != -1:
                    values = line.split(':')[1].strip().split(',')
                    values = [x.strip() for x in values]
                    values = list(filter(None, values))
                    if discrete_column_values is None:
                        discrete_column_values = {}
                    discrete_column_values[column_name] = values
        else:   
            print("[-] Error: unknown line: " + line)
            continue
            
 
             
    ''' parse columns: '''
    if target_column is None:
        column_names = [line.split(':')[0] for line in content]
        column_names = list(filter(None, column_names))
        column_names = [re.sub(r':$', '', x) for x in column_names]
        data_types = [line.split(':')[1] for line in content]
        data_types = list(filter(None, data_types))


    
    
    ''' load training data '''
    '''pd.read_csv needs to ignore lines that starts with comment like #,|,!, etc. :'''
    
    
    df_train = pd.read_csv(datafile, sep=',', header=None, index_col=None,names=column_names,skip_blank_lines=True)
    df_test = pd.read_csv(testfile, sep=',', header=None, index_col=None,names=column_names)
    '''sanitize df_train and df_test''' 
    df_test = df_test.dropna()
    df_train = df_train.dropna()
    
    print(column_names)
    print(data_types)
    print(target_column)
    
    df_train[target_column] = df_train[target_column].astype('category')
    df_test[target_column] = df_test[target_column].astype('category')
    
    
    if discrete_column_values is not None:
        for column_name in discrete_column_values:
            df_train[column_name] = df_train[column_name].astype('category')
            df_test[column_name] = df_test[column_name].astype('category')
            df_train[column_name].cat.set_categories(discrete_column_values[column_name])
            df_test[column_name].cat.set_categories(discrete_column_values[column_name])

    
    '''set up data types'''
    for i in range(0, len(data_types)):
        if data_types[i] == 'discrete':
            df_train[column_names[i]] = df_train[column_names[i]].astype('category')
            df_test[column_names[i]] = df_test[column_names[i]].astype('category')  
        elif data_types[i] == 'continuous' or data_types[i] == 'numeric':
            df_train[column_names[i]] = df_train[column_names[i]].astype('float64')
            df_test[column_names[i]] = df_test[column_names[i]].astype('float64')
        else:
            print("[-] Error: unknown data type: " + data_types[i])
            sys.exit(1)
    '''set up target values'''
    if target_values is not None:
        df_train[target_column].cat.set_categories(target_values)
        df_test[target_column].cat.set_categories(target_values)


    ''' remove the datafile '''
    # Return the tuple of train/test dataframes
    return (df_train, df_test, target_column, target_values)





def benchmark_folder(folder, normalize=False):
        # Read the data
        # ===================
        classifiers = [ ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, GaussianNB, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, KNeighborsClassifier, SVC, LogisticRegression, MLPClassifier ,KNeighborsClassifier  ]  

        print("[+] iterating "+folder + " ..." )
        for file in os.listdir(folder):
            if file.endswith(".names") :
                print("[+] plotting "+file + " ..." )
                # Parse the .names file
                # ===================
                df_train, df_test, target_column, target_values = parse_names_file(file,folder)
                
                #train the model, predict the test data, and print the metrics 
                #create one plot for each file stem with results
                #fix unsupported format string passed to numpy.ndarray.__format__ 
        

                for classifier in classifiers:
                    
                    try:
                        print("Training " + classifier.__name__)
                        clf = classifier()
                        clf.fit(df_train, df_train[target_column])
                        y_pred = clf.predict(df_test)
                        print("Accuracy:",metrics.accuracy_score(df_test[target_column], y_pred))
                        print("Precision:",metrics.precision_score(df_test[target_column], y_pred, average='weighted'))
                        print("Recall:",metrics.recall_score(df_test[target_column], y_pred, average='weighted'))
                        print("F1:",metrics.f1_score(df_test[target_column], y_pred, average='weighted'))
                        print("Confusion Matrix:")
                        print(metrics.confusion_matrix(df_test[target_column], y_pred))
                        #print("Classification Report:")
                        #print(metrics.classification_report(df_test[target_column], y_pred))
                        # Plot the metrics
                        # set new plot size
                        if target_values is None:
                            target_values=df_train[target_column].unique()
                      
                        n_classes = len(target_values)
                        # set text size
                        sns.set(font_scale=1.4) # for label size
                        sns.set_style("whitegrid") 
                        fig = plt.figure(figsize=(20,30)) 
                        #make 5 subplots
                        ax = fig.add_subplot(3,2,1)
                        ax2 = fig.add_subplot( 3,2,2)
                        ax3 = fig.add_subplot(  3,2,3)
                        ax4 = fig.add_subplot(  3,2,4)

                        #add a new figure for metrics 
                        ax5 = fig.add_subplot(  3,2,5)
                        ax5_bottom = ax5#ax5.inset_axes([0, -0.2, 1, 0.2])

                        

                        
                        
                        #add subplots for metrics 
                        ax.set_title(classifier.__name__ + " Confusion Matrix")
                        ax.set_xlabel('Predicted labels')
                        ax.set_ylabel('True labels')
                        ax2.set_title(classifier.__name__ + " ROC Curve")
                        ax3.set_title(classifier.__name__ + " Feature Importances")
                        ax4.set_title(classifier.__name__ + " Density Plot")
                        ax2.set_xlabel('False Positive Rate')
                        ax2.set_ylabel('True Positive Rate')
                        ax2.set_xlim([0.0, 1.0])
                        ax2.set_ylim([0.0, 1.05])
                        ax2.grid(True)
                        
                        ax3.set_xlabel('Feature')
                        ax3.set_ylabel('Importance')
                        ax3.set_xlim([-1, df_train.shape[1]])
                        ax3.set_ylim([0, 1])
                        ax3.grid(True)
                        ax4.grid(True)
                        ax4.set_xlabel('Value')
                        ax4.set_ylabel('Density')
                        ax4.set_xlim([0, 1])
                        ax4.set_ylim([0, 1])
                        ax4.grid(True)
                        ax5.grid(False)
                        ax5_bottom.set_title(classifier.__name__ + " Metrics")
                      
                        #remove legends
                        ax.show_legend = False
                        ax2.show_legend = False
                        ax3.show_legend = False
                        ax4.show_legend = False
                        ax5.show_legend = False
                        ax5_bottom.show_legend = False
                        plt.show_legend = False
                        fig.show_legend = False

                        if normalize:
                            ax.set_title(classifier.__name__ + " Normalized Confusion Matrix")
                            sns.heatmap(metrics.confusion_matrix(df_test[target_column], y_pred, normalize='true'), annot=True, fmt='.2%', cmap='flare_r', ax=ax) 
                            ax.set_xlabel('Predicted labels')
                            ax.set_ylabel('True labels')
                            
                            #save the plot
                        else:
                            ax.set_title(classifier.__name__ + " Confusion Matrix")
                            sns.heatmap(metrics.confusion_matrix(df_test[target_column], y_pred), annot=True, fmt='g', cmap='flare_r', ax=ax) 
                            ax.set_xlabel('Predicted labels')
                            ax.set_ylabel('True labels')

                            #save the plot
                            
                        #add subplots for metrics
                        
                        #normalize the confusion matrix
                        #add a new subplot for the roc curve
                         #plot the roc curve with roc_auc_score
                        try:
                            colors = cycle(['blue', 'red', 'green', 'yellow', 'orange'])
                            for i, color in zip(range(n_classes), colors):
                                fpr, tpr, thresholds = metrics.roc_curve( df_test[target_column], y_pred, pos_label=i)
                                ax2.plot(fpr, tpr, color=color, lw=2,
                                        label='ROC curve of class {0} (area = {1:0.2f})'
                                        ''.format(i, metrics.auc(fpr, tpr)))
                            #metrics.plot_roc_curve(clf, df_test, df_test[target_column], ax=ax2, name=classifier.__name__ ,)
                            '''avoid    ValueError: multiclass format is not supported'''
                            '''avoid 'bool' object is not subscriptable '''

                            ax2.show_legend = False

                            #metrics.roc_auc_score(df_test[target_column], df_test[target_column])
                            #metrics.plot_roc_curve(clf, df_test, df_test[target_column], ax=ax2, name=classifier.__name__ ,)
                        except Exception as e:
                            print("[-] ROC plot error: " + str(e))

                        #add a new subplot for feature importances
                        #plot the feature importances
                        try:
                            #if classifier has feature_importances_ attribute 
                            if hasattr(clf, 'feature_importances_'):
                                importances = clf.feature_importances_
                                std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
                            else:
                                   #rank features by their importance 
                                    importances = np.zeros(df_train.shape[1])
                                    for i in range(0, df_train.shape[1]):
                                        if i==target_column:
                                            continue
                                        importances[i] = metrics.mutual_info_score(df_train[target_column], df_train.iloc[:,i])
                                    importances = importances / np.sum(importances)
                                    std = None
                                    
                        except Exception as e:
                            print("[-]Importance error: " + str(e))
                        try:
                            indices = np.argsort(importances)[::-1]
                            # Print the feature ranking
                            column_names = df_train.columns.values

                            print("Feature ranking:")
                            for f in range(df_train.shape[1]):
                                print("%d. feature %d (%s) (%f)" % (f + 1, indices[f], column_names[indices[f]] ,importances[indices[f]]))
                            # Plot the feature importances of the forest
                            hints = [column_names[indices[f]] for f in range(df_train.shape[1])]
                            yerr = std[indices] if std is not None else None
                            ax3.set_xticks(range(df_train.shape[1]))
                            ax3.set_xticklabels(hints, rotation=90)
                            ax3.set_xlabel('Feature')
                            ax3.set_ylabel('Importance')
                            ax3.set_xlim([-1, df_train.shape[1]])
                            ax3.set_ylim([0, 1])
                            ax3.grid(True)
                            ax3.bar(range(df_train.shape[1]), importances[indices], yerr=yerr, align="center")


                        except Exception as e:
                            print("[-]Importance ranking error: " + str(e))
                                 
                        #add a new subplot for density plot
                        #plot the density plot
                      
                        #ax4.grid(False)
                        #ax4.show_legend = False


                        try: 
                          #combined_summary = pd.concat([df_train,df_test] , axis=0)
                          #total_columns = combined_summary.columns
                          #num_col = combined_summary._get_numeric_data().columns
                          #cat_col = list(set(total_columns) - set(num_col))
                          #  for column in cat_col:  
                          #      sns.kdeplot(combined_summary[column], ax=ax4, shade=True, color="blue", label=column)
                          #  for column in num_col:
                          #      sns.kdeplot(combined_summary[column], ax=ax4, shade=True, color="red", label=column)
                          combined_summary = pd.concat([df_train,df_test] , axis=0) 
                          combined_numeric_summary = combined_summary._get_numeric_data()
                          for column in combined_numeric_summary.columns:
                              sns.kdeplot(combined_numeric_summary[column], ax=ax4, shade=True, label=column)  
                            
                          #set layout
                          ax4.set_xlabel('Value')
                          ax4.set_ylabel('Density')
                          ax4.set_xlim([0, 1])
                          ax4.set_ylim([0, 1])
                          ax4.grid(True)
                          ax4.show_legend = False
                        except Exception as e:
                            print("[-] density error: " + str(e))
                            
                        #save the plot
                        #done, save the plot
                        #fill ax5 with table of metrics from train and test data results :
                        try:
                            results = pd.DataFrame()
                            train_results = pd.DataFrame()
                            test_results = pd.DataFrame()
                            train_results['Accuracy'] = [metrics.accuracy_score(df_train[target_column], clf.predict(df_train))]
                            train_results['Precision'] = [metrics.precision_score(df_train[target_column], clf.predict(df_train), average='weighted')]
                            train_results['Recall'] = [metrics.recall_score(df_train[target_column], clf.predict(df_train), average='weighted')]
                            train_results['F1'] = [metrics.f1_score(df_train[target_column], clf.predict(df_train), average='weighted')]
                            test_results['Accuracy'] = [metrics.accuracy_score(df_test[target_column], clf.predict(df_test))]
                            test_results['Precision'] = [metrics.precision_score(df_test[target_column], clf.predict(df_test), average='weighted')]
                            test_results['Recall'] = [metrics.recall_score(df_test[target_column], clf.predict(df_test), average='weighted')]
                            test_results['F1'] = [metrics.f1_score(df_test[target_column], clf.predict(df_test), average='weighted')]
                            results = pd.concat([train_results, test_results], axis=0)
                            #results.suptitle(classifier.__name__ + " Metrics")

                            results.index = ['Train', 'Test']
                            results.index.name = 'Dataset'
                            results.columns.name = 'Metric'
                            sns.heatmap(results, annot=True, fmt='.2%', cmap='flare_r', ax=ax5_bottom)
                            ax5_bottom.set_xlabel('Metric')
                            ax5_bottom.set_ylabel('Value')
                            ax5_bottom.set_title(classifier.__name__ + " Metrics")
                            ax5_bottom.show_legend = False
 
                            
                        except Exception as e:
                            print("[-] results table error: " + str(e))
                        #save the plot
 
                        print("[+]Saving figure to ./" + file.split('.')[0] + "."+classifier.__name__+"_confusion.png")
                        title = (classifier.__name__ +" "+ file.split('.')[0]+" Metrics")
                        #set main title
                        fig.suptitle(title)
                        
                        plt.savefig("./" + file.split('.')[0] + "."+classifier.__name__+"_confusion.png" , dpi=300)
                        plt.close()

                   
                    except Exception as e:
                        print("[-] General Error: " + str(e))
                        plt.close()
                        continue

               
''' main '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot confusion matrices')
    parser.add_argument('-f', '--folder', type=str, help='Folder containing the .names files', required=True)
    parser.add_argument('-n', '--normalize', action='store_true', help='Normalize the confusion matrix', required=False)
    args = parser.parse_args()
    benchmark_folder(args.folder, args.normalize)