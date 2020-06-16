import os
from converter import convert_wav_file

# The source directory containing all the .wav sound files to be processed
source_directory = "c:/Users/ricoz/Desktop/Python Workspace/BeachBoysHarmonizer/assets"

# Audio source folder
audio_directory = source_directory + "/source"

# Iterate through all audio files in the audio source folder and process them
for filename in os.listdir(audio_directory):

    # Strip off the .wav ending from the filename string
    filename = filename[:-4]
    print(filename)

    # Preprocess and convert the audio .wav file to the image and save it to the spectrograms folder
    convert_wav_file(source_directory, filename)