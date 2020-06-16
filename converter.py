
'''
Converter.py

This file converts from .mp3 files to spectrograms via librosa's
Short-Term Fourier Transformations (STFT) graphing algorithms.

'''

import numpy as np

import matplotlib.pyplot as plt

import librosa
import librosa.display

# The string path to the exmaple .wav file
audio_path = "c:/Users/ricoz/Desktop/Python Workspace/BeachBoysHarmonizer/assets/data/doitagain.wav"

# Buffer the currently loaded audio file into 
y, sr = librosa.load(audio_path)

# Construct and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power (max) as reference.
log_S = librosa.power_to_db(S, ref=np.max)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()

# Display the spectrogram
plt.show()

print("A wild success!")
