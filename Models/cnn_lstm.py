import keras
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Reshape, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam


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
TRAIN_PATH = 'features/mfcc/split_train_less_unknown'
VAL_PATH = 'features/mfcc/split_validation_less_unknown'
BATCH_SIZE = 128
EPOCHS = 100
def load_data(path, dict_comm):
	dataset_x = []
	dataset_y = []
	for subdir, dirs, files in os.walk(path):
		print (subdir)
		for file in files:
			with open(os.path.join(subdir, file)) as f:
				feature_name = file.split('_')[0]
				feature_label = subdir.split('/')[-1]
				feature = np.array(f.read().replace('\n', ' ').split(' '))
				feature = np.array([x  for x in  feature if x]).astype(np.float)
				if feature_label in dict_comm.keys():
					int_feat_label = dict_comm[feature_label]
				else:
					int_feat_label = dict_comm['unknown']
				feature = feature.reshape(feature_x, feature_y, 1)
				label = [0 for _ in range(len(dict_comm))]
				label[int_feat_label] = 1
				dataset_x.append(feature)
				dataset_y.append(label)
	return np.array(dataset_x), np.array(dataset_y)



dict_commands, inv_dict = load_dictionary()
feature_x = 16
feature_y = 13

print ('Loading Training Set into memory...')
train_set_X, train_set_Y = load_data(TRAIN_PATH, dict_commands)
train_set_X = train_set_X.reshape(train_set_X.shape[0], feature_x, feature_y, 1)
print(train_set_X.shape, train_set_Y.shape)
print ('Loading Test Set int memory...')
val_set_X, val_set_Y = load_data(VAL_PATH, dict_commands)
val_set_X = val_set_X.reshape(val_set_X.shape[0], feature_x, feature_y, 1)
print(val_set_X.shape, val_set_Y.shape)

model = Sequential()


#16x13x1

model.add(Conv2D(256, kernel_size = (5,5), input_shape = (feature_x, feature_y, 1), padding = 'same'))
model.add(MaxPooling2D(pool_size = (5, 5), strides = (1,1), padding = 'same'))
model.add(BatchNormalization())
#16x13x256


model.add(Conv2D(128, kernel_size = (3,2), activation = 'relu'))
model.add(BatchNormalization())
#14x12x128

model.add(Conv2D(64, kernel_size = (2,2)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (1,1)))
model.add(BatchNormalization())
#12x10x64


model.add(Conv2D(1, kernel_size = (3,3), padding = 'same'))
#12x10x1

model.add(Reshape((12, 10)))

model.add(LSTM(256, return_sequences = True))
model.add(LSTM(256, return_sequences = True))

model.add(Dense(128))

model.add(TimeDistributed(Dense(64)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(len(dict_commands)))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss="categorical_crossentropy", optimizer = adam, metrics = ['accuracy'])
model.summary()

history = LossHistory()

for nb_epoch in range(EPOCHS):
	print ('\n\nEpoch: {}\n'.format(nb_epoch))
	model.fit(train_set_X, train_set_Y, batch_size = BATCH_SIZE, verbose = 1, nb_epoch = 1, callbacks = [history], validation_data = (val_set_X, val_set_Y))
	lossFile = open("losses.txt", "a")
	lossFile.write(str(history.losses))
	lossFile.close()
	print ('Validation Set Accuracy: ')
	print (model.evaluate(val_set_X, val_set_Y))
	print ('\n\n')
	if nb_epoch % 10 == 0:
		model.save_weights('checkpoint_epoch_{}.hdf5'.format(nb_epoch))

predicted = model.predict(val_set_X)	
with open('submission_validation.csv', 'w+') as f:
	f.write('label_true,label_predicted\n')
	for i in range(len(val_set_X)):
		f.write(val_set_Y[i] + '.wav,' + inv_dict[np.argmax(predicted[i])] + '\n')

