import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split as split
import matplotlib
import keras
import os
from keras.layers import *
from keras.models import Model
from keras.layers import CuDNNLSTM, Bidirectional, TimeDistributed, Activation, Dot, Permute, Flatten, add
from keras import regularizers
from keras.optimizers import SGD,Adam,RMSprop
from keras.optimizers import Adam
from keras import backend as K
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import preprocessing
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
TRAIN_PATH = 'features/mel_power/split_train_less_unknown'
VAL_PATH = 'features/mel_power/split_validation_less_unknown'


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
        input_length = np.zeros(self.batch_size)
        for i in range(batch_size):
            input_length[i] = 16
        label_len = []
        for element in batch_y:
            label = inv_dict[np.argmax(element)]
            label_len.append(len(label))
        label_length = np.array(label_len)

        inputs = {
            'input_1': batch_x,
            'the_labels': batch_y,
            'input_length': input_length,
            'label_length': label_length
        }

        outputs = {'ctc': np.zeros([self.batch_size])}
        return (inputs, outputs)

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




def load_data(path, dict_comm):
    dataset_x = []
    dataset_y = []
    for subdir, dirs, files in os.walk(path):
        print (subdir.split('/')[-1])
        for file in files:
            with open(os.path.join(subdir, file)) as f:
                feature_name = file.split('_')[0]
                feature_label = subdir.split('/')[-1]
                feature = np.array(f.read().replace('\n', ' ').split(' '))
                feature = np.array([x  for x in  feature if x]).astype(np.float)
                feature = np.nan_to_num(feature)
                if feature_label in dict_comm.keys():
                    int_feat_label = dict_comm[feature_label]
                else:
                    int_feat_label = dict_comm['unknown']
                feature = feature.reshape(feature_x, feature_y, 1)
                label = [0 for _ in range(len(dict_comm))]
                label[int_feat_label] = 1
                dataset_x.append(feature)
                dataset_y.append(label)
                #return np.array(dataset_x), np.array(dataset_y)
    return np.array(dataset_x), np.array(dataset_y)


# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    # hack for load_model
    import tensorflow as tf

    ''' from TF: Input requirements
    1. sequence_length(b) <= time for all b
    2. max(labels.indices(labels.indices[:, 1] == b, 2)) <= sequence_length(b) for all b.
    '''

    # print("CTC lambda inputs / shape")
    # print("y_pred:",y_pred.shape)  # (?, 778, 30)
    # print("labels:",labels.shape)  # (?, 80)
    # print("input_length:",input_length.shape)  # (?, 1)
    # print("label_length:",label_length.shape)  # (?, 1)


    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc(y_true, y_pred):
    return y_pred



def new_model():  
    def Penalization(matrix):
        return K.sqrt(K.sum(K.square(K.abs(matrix - K.eye(16)))))

    inputs = Input(shape=(16,128,1))
    
    conv = Conv2D(32, 3, activation='relu', padding = 'same', kernel_regularizer=regularizers.l2(0.001))(inputs)
    conv = BatchNormalization()(conv)  
    
    conv1 = MaxPooling2D((1, 2))(conv)
    conv1 = concatenate([conv1, MaxPooling2D((1, 2))(Conv2D(1,1, activation='relu',padding = 'same')(inputs))])
    #16 x 64 x 16
    
    conv = Conv2D(32, 3, activation='relu', padding = 'same', kernel_regularizer=regularizers.l2(0.001))(conv1)
    conv = BatchNormalization()(conv)
   
    conv2 = MaxPooling2D((1, 2))(conv)
    conv2 = concatenate([conv2, MaxPooling2D((1, 2))(Conv2D(1,1,activation='relu',padding = 'same')(conv1))])
    #16 x 32 x 32
    
    #conv = AveragePooling2D((1, 32))(conv)
    conv = Conv2D(32, 3, activation='relu', padding = 'same', kernel_regularizer=regularizers.l2(0.001))(conv2)
    conv = BatchNormalization()(conv)

    conv3 = MaxPooling2D((1, 2))(conv)
    conv3 = concatenate([conv3, MaxPooling2D((1, 2))(Conv2D(1,1,activation='relu',padding = 'same')(conv2))])
    #conv = Reshape((16, 1024))(conv)
    #16 x 16 x 32
    
    
    conv = Reshape((16, 528))(conv3) #32 tempos e 32 filtros em 4 posicoes contatenados

    H = Bidirectional(CuDNNLSTM(int(num_units), return_sequences=True, name='H', kernel_regularizer=regularizers.l2(0.01)))(conv)
    H = Bidirectional(CuDNNLSTM(int(num_units), return_sequences=True, name='H', kernel_regularizer=regularizers.l2(0.01)))(H)
    #H = Bidirectional(CuDNNGRU(int(num_units), return_sequences=True, name='H', kernel_regularizer=regularizers.l2(0.01)))(H)
    #H = Bidirectional(CuDNNGRU(int(num_units), return_sequences=True, name='H', kernel_regularizer=regularizers.l2(0.01)))(H)
    

    Ws1 = Dense(latent_dim, activation='tanh', use_bias=False, name='Ws1')(H)
    Ws2 = Dense(latent_atten, activation='linear', use_bias=False, name='Ws2')(Ws1)

    A = TimeDistributed(Activation('softmax'), name='Softmax')(Permute((2,1))(Ws2))
    
    A_q = Dot((1,2))([A,K.permute_dimensions(A,(0,2,1))])
    
    P = Penalization(A_q)
    
    atten_model = Model(inputs=inputs, outputs=A)
    
    M = Flatten()(dot([A, H],(2,1)))
    
    M = Dropout(0.2)(M)

    y_pred = Dense(12, activation='softmax', name = "y_pred")(M)
    y_pred = A


    
    def custom_loss(y_true,y_pred):  
        loss = K.categorical_crossentropy(y_pred, y_true)  
        return loss + P*0.1

     # Change shape
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')


    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    model =  Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
    model.compile(optimizer=Adam(0.001, decay=0.000001),loss=ctc, metrics=['acc'])
    model.summary()
    
    return model, atten_model

#--------------------------MAIN----------------------------------#
dict_commands, inv_dict = load_dictionary()
feature_x = 16
feature_y = 128

print ('Loading Training Set into memory...')
train_set_X, train_set_Y = load_data(TRAIN_PATH, dict_commands)
train_set_X = train_set_X.reshape(train_set_X.shape[0], feature_x, feature_y, 1)
print(train_set_X.shape, train_set_Y.shape)
print ('Loading Test Set int memory...')
val_set_X, val_set_Y = load_data(VAL_PATH, dict_commands)
val_set_X = val_set_X.reshape(val_set_X.shape[0], feature_x, feature_y, 1)
print(val_set_X.shape, val_set_Y.shape)

num_units = 32
latent_atten = 16
latent_dim = 64

model, atten_model = new_model()


epocas = 1
batch_size = 128

loss = []
acc = []

train_data = BatchGenerator(train_set_X, train_set_Y, 128, True)

valid_data = BatchGenerator(val_set_X, val_set_Y, 128, True)

train_steps = len(train_set_X) // batch_size

valid_steps = len(val_set_X) // batch_size


for e in range(epocas):
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
    
    #print('ACC', model.evaluate(X_test[...,np.newaxis], y_test,verbose=0)[1])
    #y_test_ind = np.argmax(y_test, axis=1)
    #y_pred = model.predict(X_test[...,np.newaxis])
    #y_pred = model.predict(X_test[...,np.newaxis])
    #y_pred_ind = np.argmax(y_pred, axis=1)
    #cnf_matrix = np.nan_to_num(confusion_matrix(y_test_ind, y_pred_ind))
    
    #plt.figure(figsize=(10,10))
    #plot_confusion_matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')
    #plt.show()
model.save_weights('checkpoint_epoch_{}.hdf5'.format(1))
    
l = np.array(loss)
a = np.array(acc)

plt.plot(l[:,0])
plt.plot(l[:,1])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(a[:,0])
plt.plot(a[:,1])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
