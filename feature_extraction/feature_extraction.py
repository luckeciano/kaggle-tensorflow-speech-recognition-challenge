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


#VAD
import webrtcvad
import struct

RESAMPLED_TRAIN_PATH = '../input/resampled/'
TRAIN_PATH = '../input/train/'
SPECGRAM_PATH = '../input/spectrogram/'
MELPOWER_PATH = '../input/mel_power/'
MFCC_PATH = '../input/mfcc/'


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def get_specgram(samples, sample_rate):
	freqs, times, spectrogram = log_specgram(samples, sample_rate)
	return freqs, times, spectrogram

def print_specgram(freqs, times, spectrogram, samples, sample_rate, filename, filepath):

	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(211)
	ax1.set_title('Raw wave of ' + filename)
	ax1.set_ylabel('Amplitude')
	ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)
	
	ax2 = fig.add_subplot(212)
	ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
	           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
	ax2.set_yticks(freqs[::16])
	ax2.set_xticks(times[::16])
	ax2.set_title('Spectrogram of ' + filename)
	ax2.set_ylabel('Freqs in Hz')
	ax2.set_xlabel('Seconds')
	if not os.path.exists(os.path.dirname(filepath)):
		print (filepath)
		os.makedirs(os.path.dirname(filepath))
	fig.savefig(filepath + filename + ".png")
	plt.close()


def resample(filename):
	# 1 - Resampling - Dimensionality Reduction
	new_sample_rate = 8000.0
	sample_rate, samples = wavfile.read(filename)
	resampled = signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))
	resampled = np.asarray(resampled, dtype=np.int16)
	return resampled, int(new_sample_rate)


def voice_activity_detection(samples, sample_rate):
	vad = webrtcvad.Vad()

	# set aggressiveness from 0 to 3
	vad.set_mode(3)
	raw_samples = struct.pack("%dh" % len(samples), *samples)
	window_duration = 0.03 # duration in seconds

	samples_per_window = int(window_duration * sample_rate + 0.5)

	bytes_per_sample = 2

	segments = []

	for start in np.arange(0, len(samples), samples_per_window):
	    stop = min(start + samples_per_window, len(samples))
	    
	    is_speech = vad.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample], 
	                              sample_rate = sample_rate)

	    segments.append(dict(
	       start = start,
	       stop = stop,
	       is_speech = is_speech))
	seg = 0
	for segment in segments:
		 if segment['is_speech']:
		 	seg += 1
	speech_samples = []
	if seg is not 0:
		speech_samples = np.concatenate([ samples[segment['start']:segment['stop']] for segment in segments if segment['is_speech']])
	speech_samples = np.asarray(speech_samples, dtype=np.int16)
	return speech_samples

def pad_audio(data, fs, T):
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape    
    N_pad = N_tar - shape[0]
    print("Padding with %s seconds of silence" % str(float(N_pad)/fs) )
    shape = (N_pad,) + shape[1:]
    # Stack only if there is something to append    
    if shape[0] > 0:                
        if len(shape) > 1:
            speech_data = np.vstack((data, np.zeros(shape)))
            speech_data = np.asarray(speech_data, dtype = np.int16)
            return speech_data
        else:
            speech_data = np.hstack((data, np.zeros(shape)))
            speech_data = np.asarray(speech_data, dtype = np.int16)
            return speech_data
                            
    else:
    	data = np.asarray(data, dtype = np.int16)
        return data

def normalize_specgram(spectrogram):
	mean = np.mean(spectrogram, axis=0)
	std = np.std(spectrogram, axis=0)
	normalized_spectrogram = (spectrogram - mean) / std
	return normalized_spectrogram

def get_mel_power_specgram(samples, sample_rate):
	# From this tutorial
	# https://github.com/librosa/librosa/blob/master/examples/LibROSA%20demo.ipynb
	S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)

	# Convert to log scale (dB). We'll use the peak power (max) as reference.
	log_S = librosa.power_to_db(S, ref=np.max)
	return log_S

def print_mel_power(log_S, sample_rate, filepath, filename):
	fig = plt.figure(figsize=(12, 4))
	librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
	plt.title('Mel power spectrogram ')
	plt.colorbar(format='%+02.0f dB')
	plt.tight_layout()
	if not os.path.exists(os.path.dirname(filepath)):
		print (filepath)
		os.makedirs(os.path.dirname(filepath))
	fig.savefig(filepath + filename + ".png")
	plt.close()

def get_mfcc(log_S):
	mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

	# Let's pad on the first and second deltas while we're at it
	delta2_mfcc = librosa.feature.delta(mfcc, order=2)
	return delta2_mfcc

def print_mfcc(delta2_mfcc, filepath, filename):
	fig = plt.figure(figsize=(12, 4))
	librosa.display.specshow(delta2_mfcc)
	plt.ylabel('MFCC coeffs')
	plt.xlabel('Time')
	plt.title('MFCC')
	plt.colorbar()
	plt.tight_layout()
	if not os.path.exists(os.path.dirname(filepath)):
		print (filepath)
		os.makedirs(os.path.dirname(filepath))
	fig.savefig(filepath + filename + ".png")
	plt.close()

def feature_extraction (subdir, file):
	filename = os.path.join(subdir, file)
	final_directory_audio = RESAMPLED_TRAIN_PATH + os.path.basename(subdir) + "/"
	final_directory_specgram = SPECGRAM_PATH + os.path.basename(subdir)  + "/"
	final_directory_melpower = MELPOWER_PATH + os.path.basename(subdir)  + "/"
	final_directory_mfcc = MFCC_PATH + os.path.basename(subdir)  + "/"
	
	#1 - Resample: Dimensionality Reduction
	resampled, new_sample_rate = resample(filename)	
	
	#OBS: Este primeiro padding deixa todos os audios com 1 segundo de duracao. Se nao fizer esse padding,
	#haverao audios que o VAD nao conseguira processar
	resampled = pad_audio(resampled, new_sample_rate, 1.0)
	
	#2 - Voice Activity Detection
	resampled_vad = voice_activity_detection(resampled, new_sample_rate)

	#3 - Padding with zeroes
	resampled_vad_padded = pad_audio(resampled_vad, new_sample_rate, 1.0)

	#4 - Feature 1: Spectrogram
	freqs, times, spectrogram = get_specgram(resampled_vad_padded, new_sample_rate)

	#print_specgram(freqs, times, spectrogram, resampled_vad_padded, new_sample_rate, "resampled_vad_padded", final_directory_specgram)

	norm_specgram = normalize_specgram(spectrogram)

	print_specgram(freqs, times, norm_specgram, resampled_vad_padded, new_sample_rate, file, final_directory_specgram)

	#5 - Feature 2: Mel Power Spectrogram
	mel_power_specgram = get_mel_power_specgram(resampled_vad_padded, new_sample_rate)

	#6 - Feature 3: MFCC
	mfcc = get_mfcc(mel_power_specgram)
	
	print_mel_power(mel_power_specgram, new_sample_rate, final_directory_melpower, file)
	
	print_mfcc (mfcc, final_directory_mfcc, file)

	if not os.path.exists(os.path.dirname(final_directory_audio)):
		print (final_directory_audio)
		os.makedirs(os.path.dirname(final_directory_audio))
	wavfile.write(os.path.join(final_directory_audio, file), int(new_sample_rate), resampled_vad_padded)
	print (os.path.join(final_directory_audio, file))




#feature_extraction("../input/train/audio/seven/", "b9f46737_nohash_1.wav")


for subdir, dirs, files in os.walk(TRAIN_PATH):
	for file in files:
		if file.endswith(".wav"):
			print (file)
			feature_extraction(subdir, file)
			


