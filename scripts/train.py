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
from keras.callbacks import TensorBoard
from functions import sliding_window, windowize, build_model


metadata = pd.read_table('meta.csv')
metadata = metadata.sample(frac=1)
le = LabelEncoder()
le.fit(np.squeeze([metadata['class']]))

msk = np.random.rand(len(df)) < 0.8

train_meta = metadata[msk]
test_meta = metadata[~msk]


(train, n) = windowize(train_meta)
train_meta = train_meta.reindex(np.repeat(df.index.values, n), method='ffill')

train_labels_num = le.transform(np.squeeze([train_meta['class']]))
train_labels = to_categorical(train_labels_num, num_classes=15)

test = windowize(test_meta)
test_meta = test_meta.reindex(np.repeat(df.index.values, n), method='ffill')

test_labels_num = le.transform(np.squeeze([test_meta['class']]))
test_labels = to_categorical(test_labels_num, num_classes=15)

del metadata
del train_meta
del test_meta

batch_size = 10
(n_samples, timesteps, features) = train.shape

model = build_model(batch_input_shape, windows, timesteps, features)
model.summary()

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=10, write_graph=True, write_images=True)

 model.fit(train, train_labels, epochs=50, batch_size=10,
                  callbacks=[tensorboard])

 scores = model.evaluate(test, test_labels)
 print "Accuracy: %.2f" % (scores[1]*100.0)

 model.save_weights('run1.h5')
