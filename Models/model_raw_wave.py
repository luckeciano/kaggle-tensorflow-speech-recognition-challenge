import keras
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, AveragePooling1D, Reshape
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras import backend as K

import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa
from sklearn.utils import shuffle
import argparse


# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-WEIGHTS', type=bool, default='')
args = vars(ap.parse_args())

WEIGHTS = args['WEIGHTS']

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def load_dictionary():
	dict_commands = {'yes': 0, 'no': 1, 'up':2, 'down': 3, 'left': 4, 'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9, 'silence': 10, 'unknown': 11}
	inv_dict = {0: 'yes', 1 : 'no', 2: 'up', 3 : 'down', 4: 'left', 5: 'right', 6: 'on', 7: 'off', 8: 'stop', 9: 'go', 10: 'silence', 11: 'unknown'}
	return dict_commands, inv_dict

#TRAIN_PATH = '/augmented/audio/mfcc/split_train'
#VAL_PATH = '/augmented/audio/mfcc/split_validation'
TRAIN_PATH = 'raw_wave/split_train'
VAL_PATH = 'raw_wave/split_validation'
BATCH_SIZE = 128
EPOCHS = 100
def load_data(path, dict_comm):
	dataset_x = []
	dataset_y = []
	max_element = -10000000
	min_element = 10000000
	for subdir, dirs, files in os.walk(path):
		print (subdir)
		for file in files:
			sample_rate, samples = wavfile.read(os.path.join(subdir,file))
			feature_label = subdir.split('/')[-1]
			#eature_label = feature_label[-4]
			feature = np.array(samples).astype(np.float)
			feature = np.lib.pad(feature, (0, 16000-feature.shape[0]), 'constant', constant_values = (0))
			resample = signal.resample(samples, 8000)
		
			feature = resample.reshape((8000, 1))
			feature = np.asarray(feature, dtype = np.int16)
			if feature_label in dict_comm.keys():
					int_feat_label = dict_comm[feature_label]
			else:
				int_feat_label = dict_comm['unknown']
			label = [0 for _ in range(len(dict_comm))]
			label[int_feat_label] = 1
			dataset_x.append(feature)
			dataset_y.append(label)
	res_x = np.array(dataset_x)
	res_x = (res_x - np.mean(res_x))/np.std(res_x)
	res_y = np.array(dataset_y)
	#print(res_x)
	return res_x, res_y




class BatchGenerator(object):
    def __init__(self, training_x, training_y, batch_size, shuffling):
        self.training_x = training_x
        self.training_y = training_y
        self.batch_size = batch_size
        self.cur_index = 0
        self.shuffling = shuffling

    def get_batch(self, idx):
        batch_x = self.training_x[idx * self.batch_size:(idx+1) * self.batch_size]
        batch_y = self.training_y[idx * self.batch_size:(idx+1) * self.batch_size]

        return (batch_x, batch_y)

    def next_batch(self):
        while 1:
            assert(self.batch_size <= len(self.training_x))

            if (self.cur_index + 1) * self.batch_size >= len(self.training_x) - self.batch_size:
                self.cur_index = 0

                if(self.shuffling==True):
                    print("SHUFFLING as reached end of data")
                    self.genshuffle()

            try:
                ret = self.get_batch(self.cur_index)
            except:
                print("data error - this shouldn't happen - try next batch")
                self.cur_index += 1
                ret = self.get_batch(self.cur_index)

            self.cur_index += 1

            yield ret

    def genshuffle(self):
        self.training_x, self.training_y = shuffle(self.training_x, self.training_y)



dict_commands, inv_dict = load_dictionary()

print ('Loading Training Set into memory...')
train_set_X, train_set_Y = load_data(TRAIN_PATH, dict_commands)
#print(train_set_X.shape, train_set_Y.shape)
print ('Loading Test Set int memory...')

val_set_X, val_set_Y = load_data(VAL_PATH, dict_commands)
#print(val_set_X.shape, val_set_Y.shape)

model = Sequential()

model.add(Conv1D(8, kernel_size = 3, strides = 1, activation = 'relu', input_shape = (8000,1)))	
model.add(BatchNormalization())
model.add(Conv1D(8, kernel_size = 3, strides = 1, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2, strides = 2))

model.add(Conv1D(16, kernel_size = 3, strides = 1, activation = 'relu'))	
model.add(BatchNormalization())
model.add(Conv1D(16, kernel_size = 3, strides = 1, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2, strides = 2))

#model.add(Dropout(0.1))
model.add(Conv1D(32, kernel_size = 3, strides = 1, activation = 'relu'))	
model.add(BatchNormalization())
model.add(Conv1D(32, kernel_size = 3, strides = 1, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2, strides = 2))

#model.add(Dropout(0.1))
model.add(Conv1D(64, kernel_size = 3, strides = 1, activation = 'relu'))	
model.add(BatchNormalization())
model.add(Conv1D(64, kernel_size = 3, strides = 1, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2, strides = 2))

#model.add(Dropout(0.1))
model.add(Conv1D(128, kernel_size = 3, strides = 1, activation = 'relu'))	
model.add(BatchNormalization())
model.add(Conv1D(128, kernel_size = 3, strides = 1, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2, strides = 2))

#model.add(Dropout(0.2))
model.add(Conv1D(256, kernel_size = 3, strides = 1, activation = 'relu'))	
model.add(BatchNormalization())
model.add(Conv1D(256, kernel_size = 3, strides = 1, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2, strides = 2))

#model.add(Dropout(0.2))
model.add(Conv1D(256, kernel_size = 3, strides = 1, activation = 'relu'))	
model.add(BatchNormalization())
model.add(Conv1D(256, kernel_size = 3, strides = 1, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2, strides = 2))

#model.add(Dropout(0.2))
model.add(Conv1D(512, kernel_size = 3, strides = 1, activation = 'relu'))	
model.add(BatchNormalization())
model.add(Conv1D(512, kernel_size = 3, strides = 1, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2, strides = 2))

#model.add(Dropout(0.2))
model.add(Conv1D(512, kernel_size = 3, strides = 1, activation = 'relu'))	
model.add(BatchNormalization())
model.add(Conv1D(512, kernel_size = 3, strides = 1, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2, strides = 2))

#model.add(Dropout(0.2))
model.add(Conv1D(512, kernel_size = 3, strides = 1, activation = 'relu'))	
model.add(BatchNormalization())
model.add(Conv1D(512, kernel_size = 3, strides = 1, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2, strides = 2))

model.add(AveragePooling1D(pool_size = 1))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(12))
model.add(Activation('softmax'))
model.summary()
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss="categorical_crossentropy", optimizer = adam, metrics = ['accuracy'])

nb_epochs = 0

if WEIGHTS != '':
	model.load_weights(WEIGHTS)
	nb_epochs = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])


history = LossHistory()


loss = []
acc = []

train_data = BatchGenerator(train_set_X, train_set_Y, BATCH_SIZE, True)

valid_data = BatchGenerator(val_set_X, val_set_Y, BATCH_SIZE, True)

train_steps = len(train_set_X) // BATCH_SIZE

valid_steps = len(val_set_X) // BATCH_SIZE


for e in range(nb_epochs, EPOCHS):
    print('Epoch: ',e)
    if e == 30:
        K.set_value(model.optimizer.lr, 0.0001)
    if e == 60:
        K.set_value(model.optimizer.lr, 0.00005)
    if e == 90:
        K.set_value(model.optimizer.lr, 0.00001)
    history =  model.fit_generator(verbose=1, 
                                    generator = train_data.next_batch(),
                                    steps_per_epoch=train_steps,
                                    validation_data=valid_data.next_batch(),
                                    validation_steps=valid_steps,                                     
                                    epochs=1, 
                                    max_queue_size=10)

    loss.append([history.history['loss'][0], history.history['val_loss'][0]])
    acc.append([history.history['acc'][0], history.history['val_acc'][0]])

    if e % 10 == 0:
        model.save_weights('model_recurrent_architecture_{}.hdf5'.format(e))
    
