
'''
Converter.py

This file contains the method convert_wav_file that converts from .wav files to mel spectrograms via librosa's
Short-Term Fourier Transformations (STFT) graphing algorithms. Each audio file is first split into segments of 5 seconds each
Used to preprocess sound files to images for Convolutional Neural Networks.

'''

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import librosa
import librosa.display

# This method takes in the path to the assets folder containing the .wav audio file to be processed. It transforms the audio file into a spectrogram (and associated pandas pd array) and saved under the filename as a png in splices of 5 seconds each
def convert_wav_file(assets_path, filename):

    # Assets_path is only the path to the assets; apply pathing to the source folder and the save folder
    audio_path = assets_path + "/source/" + filename + ".wav"

    # Output path to put the generated directories
    output_path = assets_path + "/output/" + filename

    # Create the new output path
    os.makedirs(output_path)

    # Buffer the currently loaded audio file
    y, sr = librosa.load(audio_path)

    # Get the number of 5 second samples to be iterated over
    buffer = 5 * sr

    # Get the iteration operands for audio spliting
    splits_total = len(y)
    splits_written = 0
    splits_counter = 1

    while splits_written < splits_total:

        # Check to see if the buffer is exceeding total samples 
        if buffer > (splits_total - splits_written):
            buffer = splits_total - splits_written

        block = y[splits_written : (splits_written + buffer)]

        # Create a new output path for the current split
        split_path = output_path + str(splits_counter) + filename

        # Create the new split audio directory
        os.makedirs(split_path)

        # Output paths for the truncated wav file, mel spectrogram and associated array
        wav_path = split_path + "/" + filename + ".wav"
        mel_path = split_path + "/" + filename + ".png"
        pd_arr_path = split_path + "/" + filename + ".csv"

        # Write 5 second segment
        librosa.output.write_wav(wav_path, block, sr)

        # Create and populate the sub-directory
        convert_split_file(wav_path, mel_path, pd_arr_path)

        # Increment the current buffer
        splits_counter += 1
        splits_written += buffer

    print("A wild success!")

# This method reads the saved audio splice in wav_path and converts it to a mel spectrogram and associate array
def convert_split_file (wav_path, mel_path, pd_arr_path):

    # Load the new spliced audio file
    y, sr = librosa.load(wav_path)

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
