
'''
Converter.py

This file contains the method convert_wav_file that converts from .wav files to mel spectrograms via librosa's
Short-Term Fourier Transformations (STFT) graphing algorithms. Used to preprocess sound files to images for
Convolutional Neural Networks.

'''

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import librosa
import librosa.display

# This method takes in the path to the assets folder containing the .wav audio file to be processed. It transforms the audio file into a spectrogram (and associated pandas pd array) and saved under the filename as a png
def convert_wav_file(assets_path, filename):

    # Assets_path is only the path to the assets; apply pathing to the source folder and the save folder
    audio_path = assets_path + "/source/" + filename + ".wav"

    # Output paths for the mel spectrogram and associated array
    output_path = assets_path + "/output/" + filename
    mel_path = output_path + "/spectrograms/" + filename + ".png"
    pd_arr_path = output_path + "/pd_arr/" + filename + ".csv"

    # Create a new output path
    os.makedirs(output_path)

    # Buffer the currently loaded audio file
    y, sr = librosa.load(audio_path)

    # Construct and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    # Save the log_S mel spectrogram array as a seperate file under the same folder as the audio file output
    np.savetxt(pd_arr_path, log_S, delimiter=",")

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
    # plt.show()

    # Save the spectrogram in the path folder for the spectrograms as .png's
    plt.savefig(mel_path)

    print("A wild success!")
