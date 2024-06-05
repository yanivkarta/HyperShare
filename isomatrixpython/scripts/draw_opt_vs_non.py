'''
Draw the .DAT file of the optimization and non-optimization 
columns of the data file : 
1. iteration number
2. optimized time to do the operation
3. non-optimized time to do the operation 
4. operation type (draw style and color) 
'''


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os
import sys
import pandas as pd 
import seaborn as sns

# Read the data file
data = pd.read_csv('OPT_VS_NON_matrix.dat', sep=' ', header=None, names=['iteration', 'optimized_time', 'non_optimized_time', 'operation_type'])    

 


print (" optimized sum: %.2f , non optimized sum %.2f" %(data['optimized_time'].sum(),data['non_optimized_time'].sum()))



# Draw the data
#plot optimal (iteration, optimized_time ) vs non-optimal (iteration, non_optimized_time) with style and color according to the operation type 
#define style for each operation type 

#plot aggregated optimized and non-optimized time with line plot 


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)


plt.style.use('seaborn-darkgrid')


#plot aggregated optimized and non-optimized time with line plot 
sns.lineplot(data['iteration'], data['optimized_time'], label='optimized', color='red',ax=ax)
sns.lineplot(data['iteration'], data['non_optimized_time'], label='non-optimized', color='blue',ax=ax ) 

# Set the title and labels

plt.grid()
plt.tight_layout()
plt.title('Optimized vs Non-Optimized Time')
plt.xlabel('Iteration')
plt.ylabel('Time (ns)')
plt.legend()

#save the plot to a file
plt.savefig('OPT_VS_NON_matrix.png') 
# Show the plot
plt.show()
plt.close()

# Draw the data
#plot optimal (iteration, optimized_time ) vs non-optimal (iteration, non_optimized_time) with style and color according to the operation type 
#define style for each operation type
fig = plt.figure(figsize=(10, 6)) 
ax = fig.add_subplot(111)

plt.style.use('seaborn-darkgrid')

#plot bar plot for each operation type, optimized and non-optimized time


colors = ['gray', 'black', 'blue', 'red', 'orange', 'yellow', 'purple', 'pink']    

for operation_type in data['operation_type'].unique():
    data_subset = data[data['operation_type'] == operation_type]
    #make 3D bar plot for each operation type 
    #set alpha to 0.5 to make the plot transparent 

    ax.bar(data_subset['iteration'], data_subset['optimized_time'], label=operation_type + ' optimized', color=colors.pop(), alpha=0.5   )
    ax.bar(data_subset['iteration'], data_subset['non_optimized_time'], label=operation_type +' non-optimized', color=colors.pop(), alpha=0.5  )


# Set the title and labels
plt.title('Optimized vs Non-Optimized Time')
plt.xlabel('Iteration')
plt.ylabel('Time (ns)')
plt.legend()
plt.grid()
plt.tight_layout()
#add a plot of aggregated optimized and non-optimized time with line plot   :
plt.legend()

#save the plot to a file
plt.savefig('OPT_VS_NON_matrix_bar.png')
# Show the plot
plt.show()

