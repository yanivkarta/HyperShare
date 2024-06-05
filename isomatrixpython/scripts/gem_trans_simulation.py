# Description: This script is used to simulate the transmission of a video file over a network using a GEM encoding scheme. 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2 as cv 
from multiprocessing import Pool 

import os
import sys
import subprocess 
import time
import random
import re
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
#use scikit autoencoder
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score 
from sklearn.metrics import accuracy_score 


#parse image data 
from PIL import Image 
global old_frame    
old_frame = None
# Set the path to the video file 



filepath ='/home/kardon/Downloads/'
filename ='CVE-2022-24852.mp4'
video_file = filepath+filename  # Path to the video file 
print(video_file)

# Load the video file 
video = cv.VideoCapture(video_file) 

# Get the number of frames in the video 
num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT)) 
# Get the video frame rate 
frame_rate = int(video.get(cv.CAP_PROP_FPS)) 
# Get the audio sample rate 
audio_sample_rate = video.get(62)
# Get the audio sample width 
audio_sample_width = 2 
# Get the audio channels 
audio_channels = 2 
# Get the audio data 
audio_data = video.get(cv.CAP_PROP_FPS) 
# Get the audio data type
audio_data_type = video.get(cv.CAP_PROP_FPS) 
 
# Create a figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set up the axes
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')
ax.set_zlabel('Amplitude')

#add figure for static plot 
ax2 = fig.add_subplot(111, projection='3d') 

ax2.set_xlabel('Time') 
ax2.set_ylabel('Frequency')
ax2.set_zlabel('Amplitude')
ax2.set_title('Audiovisual Data Scatterplot') 

number_of_threads = num_frames//frame_rate+1    
print('Number of Threads:', number_of_threads) 

# Create the line object
line, = ax.plot([], [], [], lw=2)

# Define the animation function
def animate(i):
    # Get the current frame of the video
    global old_frame 
    
    
    #get frame[i] from the video
    video.set(1, i) 
    ret, frame = video.read() 

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    audio_data = video.get(58)
    audio_data_type = video.get(59)  
    
    #if old_frame is None:
    
    if old_frame is None:
        old_frame = frame
        return line,
        
    #calculate the difference between the current frame and the previous frame
    frame_diff = cv.absdiff(frame, old_frame) 
    old_frame = frame
    #get the audio data
    audio_data = video.get(58)
    audio_data_type = video.get(59)
    audio_sample_rate = video.get(62)

    audio_sample_width = 2
    audio_channels = 2
    frame_audio = audio_data/num_frames 
    frame_video = frame_diff/num_frames 

    print('Frame:', i, 'Audio Data:', audio_data, 'Audio Data Type:', audio_data_type, 'Audio Sample Rate:', audio_sample_rate, 'Audio Sample Width:', audio_sample_width, 'Audio Channels:', audio_channels)     
    #crate scatterplot of the frame with audio 
    #and video data scaled to the same range 
    
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
   
    x = np.linspace(-1*frame_audio, 2*np.pi*frame_audio, 100) 

    y = np.sin(x) 

    z = np.cos(x)  
    ax.scatter(x, y, z, c='r', marker='o')

    line.set_data(x, y) 
    line.set_3d_properties(z)   
    
    print('Frame:', i, 'Audio Data:', audio_data, 'Audio Data Type:', audio_data_type, 'Audio Sample Rate:', audio_sample_rate, 'Audio Sample Width:', audio_sample_width, 'Audio Channels:', audio_channels) 
    #draw frame on the plot 
    ax.imshow(frame_diff, cmap='gray', aspect='auto' ) 
    ax.set_title('Frame: {}'.format(i)) 
    #update ax2
    ax2.scatter(audio_data, frame, c='b', marker='x') 
    ax2.set_title('Audiovisual Data Scatterplot') 
    # Plot the audio data
    return line,


# Create the animation
#run in parallel    
#with Pool(3) as pool: 

#    pool.map(animate, range(num_frames))
#    pool.close()

 
anim = animation.FuncAnimation(fig, animate, frames =num_frames , interval=1000/frame_rate, blit=True, repeat=False ) 
anim.save('video_transmission.mp4', writer='ffmpeg', fps=frame_rate, extra_args=['-vcodec', 'libx264'])  # Save the animation as a video file  
# Show the animation
plt.show()
# Close the video file
video.release()
# Close the plot
plt.close()
# Set the path to the video file
