import os
from os.path import isdir, join
from pathlib import Path
import numpy as np
import librosa
from pydub import AudioSegment
import glob
import pdb

TRAIN_PATH = 'augmentation_example' # original files

DEFAULT_PATH = 'augmentation_example'
SAMPLE_RATE = 16000 
      
def save_audio(audio, audio_path, audio_name):
    if not os.path.exists(os.path.dirname(audio_path)):
        #print (audio_path)
        os.makedirs(os.path.dirname(audio_path))
        
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
    
def stretch(data, rate=1):
    input_length = 16000
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data

def load_audio_file(file_path):
    input_length = 16000
    data = librosa.core.load(file_path)[0] #, sr=16000
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data
    
def save_sound(audio, audio_path, audio_name, rate=SAMPLE_RATE):
    if not os.path.exists(os.path.dirname(audio_path)):
        #print (audio_path)
        os.makedirs(os.path.dirname(audio_path))

    librosa.output.write_wav(audio_path + audio_name, audio, int(rate))
    audioNorm = AudioSegment.from_file(audio_path + audio_name, format="wav")
    audioNorm = normalize_audio(audioNorm, -15)
    save_audio(audioNorm, audio_path, audio_name)    
    
    
def plot_time_series(data, name, rate=16000):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.savefig('fig_' +name+ '.png')
        
def data_augmentation(subdir, file):
    filename = os.path.join(subdir, file)
    final_directory = DEFAULT_PATH + "/" +  os.path.basename(subdir) + "/"
    data = load_audio_file(filename)   
    
    # Do the augmentation with background noite
    augment_bgNoise(subdir, file[:-4] + '_bgNoise1' + '.wav' , final_directory)
    augment_bgNoise(subdir, file[:-4] + '_bgNoise2' + '.wav' , final_directory)
    augment_bgNoise(subdir, file[:-4] + '_bgNoise3' + '.wav' , final_directory)
    
    # Adding white noise 
    data_wn = data + 0.005*(np.random.randn(len(data)))
    #plot_time_series(data_wn,'1')
    
    # Shifting the sound
    #data_roll = np.roll(data, 16000)

    #stretch
    deeper = stretch(data, 0.8)
    high_freq = stretch(data, 1.5)
    higher_freq = stretch(data, 1.8)
    
    #pitch_shift
    pitch_six_half_step  = librosa.effects.pitch_shift(data, SAMPLE_RATE, n_steps=4)
    pitch_tritone  = librosa.effects.pitch_shift(data, SAMPLE_RATE, n_steps=-6)
    pitch_quarter_tone  = librosa.effects.pitch_shift(data, SAMPLE_RATE, n_steps=-3, bins_per_octave=24)

    # Saving files
    save_sound(data_wn, final_directory, os.path.basename(filename)[:-4] + '_wn' + '.wav')
    #save_sound(data_roll, final_directory, os.path.basename(filename))
    #save_sound(data_roll, final_directory, '123_' + os.path.basename(filename), 13000)
    save_sound(deeper, final_directory, os.path.basename(filename)[:-4] + '_deep' + '.wav')
    save_sound(high_freq, final_directory, os.path.basename(filename)[:-4] + '_highFreq' + '.wav', 20000)
    save_sound(higher_freq, final_directory, os.path.basename(filename)[:-4] + '_higherFreq' + '.wav', 25000)
    save_sound(pitch_six_half_step, final_directory, os.path.basename(filename)[:-4] + '_pitch_sixHalf' + '.wav')
    save_sound(pitch_tritone, final_directory, os.path.basename(filename)[:-4] + '_pitchTritone' + '.wav')
    save_sound(pitch_quarter_tone, final_directory, os.path.basename(filename)[:-4] + '_pitch_quarterTone' + '.wav')
   
def augment_bgNoise(audio_path, audio_name, pathToSave):
    filename_tmp = audio_name.split('_')
    filename = filename_tmp[0] + '_' + filename_tmp[1] + '_' + filename_tmp[2] + '.wav'
    audio = AudioSegment.from_file(audio_path + "/" + filename, format="wav")
    audio = normalize_audio(audio, -15)
    
    small_bgnoise1 = slice_audio(bgnoise)
    lowerVol = np.random.randint(7,22,None)
    output = merge_audio(small_bgnoise1 - lowerVol, audio)
    
    lowerVol = np.random.randint(7,22,None)
    small_bgnoise2 = slice_audio(bgnoise)
    output_double = merge_audio(small_bgnoise2 - lowerVol, output)
    
    save_audio(output_double, pathToSave, audio_name)
    
# One single object with all background noise files normalized
noiseList = glob.glob("./Audio files/Dataset_speechCommands/_background_noise_/*.wav")
bgnoise = AudioSegment.from_wav(noiseList[0])
bgnoise = normalize_audio(bgnoise, -15)

for i in range(1, len(noiseList)):
    noise = AudioSegment.from_wav(noiseList[i])
    noise = normalize_audio(noise, -15)
    bgnoise = join_audio(bgnoise, noise) 
         
def main():  
    print('Starting the augmentation ...')      
    for subdir, dirs, files in os.walk(TRAIN_PATH):
        for file in files:    
            # Do the augmentation for every sound excepting for those on background_noise dir
            if file.endswith(".wav") and "_background_noise_" not in str(subdir):
                #print('SUBDIR: ' + str(subdir) )
                #print (file)
                data_augmentation(subdir, file)
                
        print('.')    
    print('Augmentation done!')

main()
