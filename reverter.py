'''
Reverter.py

This file contains the method revert_wav_file that converts from .png files (mel spectrograms) via librosa's
Short-Term Fourier Transformations (STFT) graphing algorithms to audio (.wav files).

'''

import numpy as np
import librosa

# This method takes in the path to the assets folder containing the .png mel spectrogram file to be processed. It transforms the .png file into a .wav audio file under filename
def convert_mel_file(assets_path, filename):

    # Assets_path is only the path to the assets; apply pathing to the source folder and the save folder
    mel_path = assets_path + "/spectrograms/" + filename + ".png"
    audio_path = assets_path + "/audio/" + filename + ".wav"

    # librosa.feature.inverse.mel_to_audio()
