#same as the previous one, but with a faster simulation speed 

# Description: This script is used to simulate the transmission of a video file over a network using a GEM encoding scheme. 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2 as cv 
from multiprocessing import Pool 
import ffmpeg

#fast simulation speed 
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

#use gpu for faster processing 
import tensorflow as tf 
#autoencoder model :
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses

#for WAV 
import wave 
#parse image data 
from PIL import Image 






def create_audio_autoencoder(wav_file):
  """
  Loads a WAV file and creates a convolutional autoencoder for audio data.

  Args:
      filepath: Path to the WAV file.

  Returns:
      A compiled TensorFlow model representing the audio autoencoder.
  """
  audio = wave.open(wav_file, "rb") # Open the WAV file
  num_channels = audio.getnchannels()  # Get the number of channels 
  sample_width = audio.getsampwidth()  # Get the sample width 
  frame_rate = audio.getframerate()  # Get the frame rate 
  num_frames = audio.getnframes()  # Get the number of frames 
# Print the audio properties
  print("Number of channels:", num_channels) 
  print("Sample width:", sample_width)
  print("Frame rate:", frame_rate)
  print("Number of frames:", num_frames)
# Load the audio samples
  audio.setpos(0)  # Set the position to the beginning of the file
  samples = audio.readframes(num_frames)  # Read all the frames
  # Convert the samples to a NumPy array 
  samples = np.frombuffer(samples, dtype=np.int16)
  # Normalize the samples
  samples = samples / 32768.0
  # Reshape the samples
  samples = samples.reshape((1, num_frames, num_channels))
  # Print the shape of the samples
  print("Samples shape:", samples.shape)
  # Create the autoencoder model
  model = tf.keras.Sequential([ 
    # Encoder
    tf.keras.layers.InputLayer(input_shape=(num_frames, num_channels)),
    tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(16, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    # Decoder
    tf.keras.layers.Conv1D(16, 3, activation="relu", padding="same"),
    tf.keras.layers.UpSampling1D(2),

    tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.UpSampling1D(2),
    tf.keras.layers.Conv1D(1, 3, activation="sigmoid", padding="same"),])

  model.compile(optimizer="adam", loss="mse")
  return model


#use ffmpeg to extract the frames from the video 



#parse image data 
from PIL import Image 
global old_frame



# Set the path to the video file 
filepath ='/home/kardon/Downloads/' 
filename ='CVE-2022-24852.mp4'
video_file = filepath+filename  # Path to the video file
print(video_file) 



def extract_frames(video_file):
    # Create a directory to store the frames
    os.makedirs('frames', exist_ok=True)
    # Use ffmpeg to extract the frames from the video
    subprocess.call(['ffmpeg', '-i', video_file, 'frames/%04d.png'])
    # Get the list of frames
    frames = sorted(os.listdir('frames'))
    return frames   # Return the list of frames

# extract the audio from the video file 
def extract_audio(video_file):
    # Use ffmpeg to extract the audio from the video
    subprocess.call(['ffmpeg', '-i', video_file, 'audio.wav'])
    # Load the audio file 
    audio = wave.open('audio.wav', 'rb') 
    return audio  # Return the audio file

# Use ffmpeg to extract the frames from the video 
frames = extract_frames(video_file) 
num_frames = len(frames) 
# Use ffmpeg to extract the audio from the video
audio = extract_audio(video_file) 
# Load the audio file 
# get the audio properties:
num_channels = audio.getnchannels() 
sample_width = audio.getsampwidth() 
frame_rate = audio.getframerate() 
audio_num_frames = audio.getnframes() 
# Print the audio properties 
# Get the frame rate of the video 
# Get the duration of the video
# frames per second 
fps = 30



duration = num_frames / fps
# Print the duration of the video 
print('Duration:', duration, 'seconds') 
# Get the first frame of the video 
frame = cv.imread('frames/0001.png') 
# Display the first frame of the video 
# Get the dimensions of the frame
height, width, _ = frame.shape
# Print the dimensions of the frame 
print('Dimensions:', width, 'x', height) 
# Get the size of the frame
size = os.path.getsize('frames/0001.png')
# Print the size of the frame
print('Size:', size, 'bytes')

#animate how the audio and video are transmitted 


# Create a figure and axis
fig, ax = plt.subplots()
ax2 = ax.twinx() 
#set ax to be a 3d plot 
ax = fig.add_subplot(111, projection='3d') 
# Create a bar to represent the transmission progress 
bar = sns.barplot(x=['Progress'], y=[0], ax=ax2) 

#create surface for the audio samples : 
x = np.linspace(0, duration, audio_num_frames) 
y = np.linspace(0, 1, num_channels) 
X, Y = np.meshgrid(x, y) 
Z = np.zeros(X.shape) 

# Plot the surface 
ax.plot_surface(X, Y, Z, cmap='viridis') 
# Set the title of the plot 
ax.set_title('Transmission Progress') 
# Set the labels of the plot 
ax.set_xlabel('Time (s)') 
ax.set_ylabel('Channel') 
ax.set_zlabel('Amplitude') 
# Set the limits of the plot 
ax.set_xlim(0, duration) 
ax.set_ylim(0, 1)
ax.set_zlim(-1, 1) 
#set the values of the bar 
bar.set_ylim(0, num_frames) 
# Set the labels of the bar
ax2.set_ylabel('Frame') 
# Set the title of the bar 
ax2.set_title('Transmission Progress') 
# Set the limits of the bar
ax2.set_ylim(0, num_frames) 


# fit the audio to the autoencoder model 

input_img = layers.Input(shape=(audio_num_frames, num_channels)) 
# Flatten the input 
x = layers.Flatten()(input_img) 
# Create the encoder 
x = layers.Dense(64, activation='relu')(x) 
x = layers.Dense(64, activation='relu')(x) 
x = layers.Dense(32, activation='relu')(x) 
# Create the decoder 

x = layers.Dense(audio_num_frames * num_channels, activation='tanh')(x) 
# Reshape the output 
decoded = layers.Reshape((audio_num_frames, num_channels))(x) 
# Create the autoencoder model 
video_autoencoder = models.Model(input_img, decoded) 
# Compile the autoencoder model 
video_autoencoder.compile(optimizer='adam', loss='mean_squared_error') 
# Print the summary of the autoencoder model 
video_autoencoder.summary() 

audio_autoencoder = create_audio_autoencoder('audio.wav') 
# Print the summary of the audio autoencoder model 
audio_autoencoder.summary() 
# Load the audio samples 
audio.setpos(0) 
samples = audio.readframes(audio_num_frames) 
# Convert the samples to a NumPy array 
samples = np.frombuffer(samples, dtype=np.int16)
# Normalize the samples 
samples = samples / 32768.0
# Reshape the samples
samples = samples.reshape((1, audio_num_frames, num_channels))
# Fit the audio samples to the autoencoder model 
audio_autoencoder.fit(samples, samples, epochs=100, batch_size=1) 
# Get the video frame
frame = cv.imread('frames/0001.png') 

# Create a figure and axis
fig, ax = plt.subplots(1, 3, figsize=(15, 5)) 
# Load the audio samples 

#create the animation functions : 
# Create the update function 

#get the audio sample of the frame 
def get_audio_sample(frame_index): 
    # Get the start and end time of the frame 
    start_time = frame_index / fps 
    end_time = (frame_index + 1) / fps 
    # Get the audio samples 
    audio.setpos(int(start_time * frame_rate)) 
    samples = audio.readframes(int((end_time - start_time) * frame_rate)) 
    # Convert the samples to a NumPy array 
    samples = np.frombuffer(samples, dtype=np.int16) 
    return samples  # Return the audio samples  

def get_video_frame(frame_index): 
    # Get the video frame 
    frame = cv.imread('frames/{:04d}.png'.format(frame_index + 1)) 
    # Convert the frame to grayscale 
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    # Resize the frame 
    frame = cv.resize(frame, (128, 128)) 
    # Normalize the frame 
    frame = frame / 255.0 
    return frame  # Return the video frame  

def get_autoencoder_video_prediction(frame): 
    # Reshape the frame 
    frame = frame.reshape((1, height, width, 1)) 
    # Get the autoencoder prediction 
    prediction = video_autoencoder.predict(frame) 
    # Reshape the prediction 
    prediction = prediction.reshape((height, width)) 
    # Denormalize the prediction 
    prediction = prediction * 255.0 
    # Convert the prediction to a NumPy array 
    prediction = np.array(prediction, dtype=np.uint8) 
    return prediction  # Return the autoencoder prediction  

def get_autoencoder_prediction(samples): 
    
    prediction = audio_autoencoder.predict(samples) 

    prediction = np.array(prediction, dtype=np.int16) 
    return prediction  # Return the autoencoder prediction  

def update(frame_index): 
    # Get the audio samples 
    samples = get_audio_sample(frame_index) 
    # Get the video frame 
    frame = get_video_frame(frame_index) 
    # Get the autoencoder prediction 
    audio_pred =  get_autoencoder_prediction(samples) 
    # Get the autoencoder prediction 
    video_pred =  get_autoencoder_video_prediction(frame) 

    # Update the bar
    bar.set_height(frame_index + 1)
    # Update the surface
    Z[:, :, 0] = samples
    ax.clear()
    ax.plot_surface(X, Y, Z, cmap='viridis')
    # Update the left scatter plot
    ax[0].clear()
    ax[0].scatter(samples, audio_pred, s=1)
    ax[0].set_xlabel('Audio')
    ax[0].set_ylabel('Autoencoder Prediction')

    # Update the right scatter plot 
    ax[1].clear()
    ax[1].scatter(frame, video_pred, s=1)

    ax[1].set_xlabel('Video')

    ax[1].set_ylabel('Autoencoder Prediction') 
    # Update the title of the plot
    ax[1].set_title('Frame {}'.format(frame_index + 1))
    # Update the plot
    plt.draw()
    # Return the updated plot
    return ax


    
# Create an animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
# Save the animation as a video file
ani.save('audio_video_autoencoder.mp4', writer='ffmpeg', fps=30)
# Display the animation
plt.show()
