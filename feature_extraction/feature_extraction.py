import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display




# 1 - Resampling - Dimensionality Reduction
RESAMPLED_TRAIN_PATH = '../input/resampled'
TRAIN_PATH = '../input/train'

for subdir, dirs, files in os.walk(TRAIN_PATH):
	for file in files:
		if file.endswith(".wav"):
			filename = os.path.join(subdir, file)
			#print(filename)
			#print(subdir)
			new_sample_rate = 8000.0
			sample_rate, samples = wavfile.read(filename)
			resampled = signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))
			resampled = np.asarray(resampled, dtype=np.int16)
			final_directory = RESAMPLED_TRAIN_PATH + subdir[2:] + "/"
			if not os.path.exists(os.path.dirname(final_directory)):
				print (final_directory)
				os.makedirs(os.path.dirname(final_directory))
			wavfile.write(os.path.join(final_directory, file), int(new_sample_rate), resampled)
			print (os.path.join(final_directory, file))

