import os
from pathlib import Path
import numpy as np
from pydub import AudioSegment
import glob
import argparse

# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-times', type=int, default=1)
ap.add_argument('-DEST_PATH', type=str, default='Audio files/Dataset_speechCommands')
args = vars(ap.parse_args())

times = args['times']
DEST_PATH = args['DEST_PATH']

SAMPLE_RATE = 16000 
      
def save_audio(audio, audio_path, audio_name):
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
        
    audio.export(audio_path + "/" + audio_name, format='wav')
    
def merge_audio(sound1, sound2):
    output = sound1.overlay(sound2, position=0)   
    return output
    
# Join two audios in sequence
def join_audio(audio1, audio2):
    joined = (audio1 + audio2)
    return joined
    
# Take a random 1s of audio
def slice_audio(audio):
    maxTime = int(len(audio)/1000) -1
    frame = (np.random.randint(0,maxTime,None))*1000
    slice = audio[frame:(frame + 1000)]
    return slice

def normalize_audio(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)
 
# One single object with all background noise files normalized
noiseList = glob.glob("./Audio files/Dataset_speechCommands/_background_noise_/*.wav")
bgnoise = AudioSegment.from_wav(noiseList[0])
bgnoise = normalize_audio(bgnoise, -15)

for i in range(1, len(noiseList)):
    noise = AudioSegment.from_wav(noiseList[i])
    noise = normalize_audio(noise, -15)
    bgnoise = join_audio(bgnoise, noise) 

def createSilence(audio_path, times=1):
	output = 0
	
	for i in range(1,times+1):
		small_bgnoise1 = slice_audio(bgnoise)
		lowerVol = np.random.randint(15,35,None)
		small_bgnoise1 = small_bgnoise1 - lowerVol
		
		lowerVol = np.random.randint(7,22,None)
		small_bgnoise2 = slice_audio(bgnoise)
		output = merge_audio(small_bgnoise2 - lowerVol, small_bgnoise1)
	
		output = normalize_audio(output, -15)
	
		save_audio(output, audio_path + '/silence', 'silence_' + str(i) + '.wav')
	
		
	
createSilence(DEST_PATH, times)

