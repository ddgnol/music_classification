import os
import scipy.io.wavfile as wav
from os import path
from pathlib import Path

# from pydub import AudioSegment
#
# BASE_DIR = Path(__file__).resolve().parent
# # files
# src = "2.mp3"
# dst = src.split(".")[0] + ".wav"
#
# # convert wav to mp3
# sound = AudioSegment.from_mp3(src)
# sound.export(dst, format="wav")

# # import required modules
# import subprocess
#
# # convert mp3 to wav file
# subprocess.call(['ffmpeg', '-i', '2.mp3',
#                  'converted_to_wav_file.wav'])
from python_speech_features import mfcc
import numpy as np
arr = np.array()
(rate, sig) = wav.read("blues.00012.wav")
mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
print(len(mfcc_feat))
covariance = np.cov(np.matrix.transpose(mfcc_feat))
print(len(covariance), covariance)
mean_matrix = mfcc_feat.mean(0)
# print(mean_matrix)
feature = (mean_matrix, covariance, 1)
# print(feature)
