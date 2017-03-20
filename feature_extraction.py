%%cython -a

import wavio
import scipy.io as sio
import librosa
import librosa.display
import pandas as pd
import numpy as np

def extract_feature_raw(file_name):
    audio = wavio.read(file_name)
    (_, channel) = audio.data.shape
    result = []
    for ch in range(channel):
        
        mfccs = librosa.feature.mfcc(y=audio.data[:,ch].astype('float32'), sr=audio.rate, n_mfcc=40)
        delta1 = librosa.feature.delta(mfccs, width=3, order=1)
        delta2 = librosa.feature.delta(mfccs, width=3, order=2)
        
        result.append(np.vstack([mfccs, delta1, delta2]))
        
    return np.vstack(result)

def extract_feature_spectrogram(file_name):
    audio = wavio.read(file_name)
    (_, channel) = audio.data.shape
    result = []
    for ch in range(channel):
        
        log_S = librosa.logamplitude(librosa.feature.melspectrogram(audio.data[:,ch].astype('float32'), sr=audio.rate), ref_power=np.max)
        mfccs = librosa.feature.mfcc(S=log_S, sr=audio.rate, n_mfcc=40)
        delta1 = librosa.feature.delta(mfccs, width=3, order=1)
        delta2 = librosa.feature.delta(mfccs, width=3, order=2)
        
        result.append(np.vstack([mfccs, delta1, delta2]))
        
    return np.vstack(result)

df = pd.read_table("/home/asura/Downloads/acoustic_modelling/TUT-acoustic-scenes-2016-development.meta/TUT-acoustic-scenes-2016-development/meta.csv")

dataset1 = np.zeros((len(df), 2584, 240))
for index, row in df.iterrows():
    #print "Processing: %s" % (row['name'])i
    features = (extract_feature_raw('../'+row['name'])).T
    dataset1[index, :,:] = features
    
    
print "Finished!"
print dataset1.shape
sio.savemat('dataset1.mat', {'data': dataset1})
