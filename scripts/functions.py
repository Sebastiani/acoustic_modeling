import wavio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation, Input
from keras.layers.advanced_activations import LeakyReLU



def sliding_window(signal, length, stride):
    (n_samples, n_channels) = signal.shape
    result = []
    
    for t in range(n_samples):
        step = stride*t
        try:
            window = signal[step: step +length, :]
            result.append(window)
        except IndexError:
            if step < n_samples:
                pad_length = n_samples - step
                pad = np.zeros((int(pad_length), n_channels))
                window = np.concatenate((signal[step:-1, :], pad), axis=0)
                result.append(window)
            break
    result = np.dstack(result, axis=3)
    return result
                
    

def create_time_windows(signal, sampling_freq, time_window):
    """
    INPUTS:  signal -> np.array of size (n_samples, n_channels) containing interesting sequence
             sampling_freq -> np.int that is Frequency used for sampling the signal in (samples/s or Hz)
             time_window -> np.int that is the time length in seconds of the signal windows
    """          
    (n_samples, n_channels) = signal.shape
    window_size = round(sampling_freq*time_window)
    num_windows = round((n_samples/window_size))
    pad_size = n_samples - num_windows*window_size
    difference = abs(window_size - pad_size)
    if difference < 100:
        
        padding = np.zeros((int(pad_size), n_channels))
        signal = np.concatenate((signal, padding), axis=0)
        print signal.shape
        return np.vstack(np.array_split(signal, int(num_windows)))
    else:
        signal = signal[0:n_samples-int(pad_size),:]
        signal = np.array_split(signal, int(num_windows))
        #print signal.shape
        #print np.stack(signal, axis=0).shape
        return np.stack(signal, axis=0)
    
def windowize(df, sampling_freq=44100, time_window=3):
    result = np.array([])
    
    for index, row in df.iterrows():
        df = df.reindex(np.repeat(df.index.values, df['N']), method='ffill')
        #filename = row['name'].split('/')
        signal = wavio.read('../'+row['name'])
        sig = create_time_windows(signal.data, sampling_freq, time_window)
        (n_labels,_,_) = sig.shape
        
        #print sig.shape
        #sig = sliding_window(signal.data, 10000, 2000)
        if result.size == 0:
            result = sig
        else:
            result = np.concatenate((result, sig), axis=0)
    #result = np.stack(result, axis=2)    
    #result = np.swapaxes(result, 0,1)
    #result = np.swapaxes(result, 0, 3)
    #result = np.swapaxes(result, 1, 2)
    #result = np.swapaxes(result, 3, 2)
    
    return (result, n_results)

def build_model(batch_size=30, windows=10, timesteps=120000, features=2):
    
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, stateful=True), merge_mode='sum', batch_input_shape=(batch_size, timesteps, features)),
        LSTM(50, return_sequences=False, stateful=True),
        Dense(30, kernel_initializer='glorot_uniform'),
        LeakyReLU(alpha=0.01),
        Dense(15, kernel_initializer='glorot_uniform'),
        Activation('softmax')
    ])

    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])
    
    return model

