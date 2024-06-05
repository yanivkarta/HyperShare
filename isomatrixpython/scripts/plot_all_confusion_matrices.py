#!/usr/bin/python3

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
import matplotlib as mpl
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
# ====================
# ====================


# Functions
  
def plot_local_folder(normalize=False):
    # Read the data
    # ===================
    local_dir = os.path.dirname(os.path.realpath(__file__)) 
    print("[+] iterating "+local_dir + " ..." )

    for file in os.listdir(local_dir):
        if file.endswith(".csv") :
            print("[+] plotting "+file + " ..." )

            df = pd.read_csv(file, sep=' ', header=None, index_col=None)
            print(df)
            # Plot the confusion matrix
            # ===================
            fig, ax = plt.subplots()
            #allocate numeric text labels for each class
            labels = [str(i) for i in range(0, len(df.columns))]
            sns.heatmap(df, annot=True, fmt='g', cmap='flare_r', ax=ax, xticklabels=labels, yticklabels=labels, cbar=False) 
            plt.tight_layout()
            print("Saving figure to " + file + ".png")
            plt.savefig(file + ".png", dpi=300)
            plt.close()
            # plt.show()

    ''' main '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot confusion matrices')
    parser.add_argument('-n', '--normalize', action='store_true', help='normalize confusion matrix')
    args = parser.parse_args()
    plot_local_folder(normalize=args.normalize)
