import wavio
import pandas as pd
import numpy as np
import keras.backend as K
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import TensorBoard
from functions import sliding_window, windowize, build_model
import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=3, allow_soft_placement=True, device_count = {'CPU': 7})
session = tf.Session(config=config)
K.set_session(session)

def replicate_func(df, n=10):
	results = [];
	for index, row in df.iterrows():
		for i in range(n):
			results.append(row['class'])
	return np.hstack(results)



metadata = pd.read_table('../meta.csv')
metadata = metadata.sample(frac=1)
metadata = metadata[0:50]
le = LabelEncoder()
le.fit(np.squeeze([metadata['class']]))

msk = np.random.rand(len(metadata)) < 0.8

train_meta = metadata[msk]
test_meta = metadata[~msk]


(train, n) = windowize(train_meta)
train_meta = replicate_func(train_meta, 10)

train_labels_num = le.transform(np.squeeze(train_meta))
train_labels = to_categorical(train_labels_num, num_classes=15)

test = windowize(test_meta)
test_meta = replicate_func(test_meta, 10)

test_labels_num = le.transform(np.squeeze(test_meta))
test_labels = to_categorical(test_labels_num, num_classes=15)

del metadata
del train_meta
del test_meta

batch_size = 10
(n_samples, timesteps, features) = train.shape
print train.shape
model = build_model(batch_size, timesteps, features)
model.summary()

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=10, write_graph=True, write_images=True)

history = model.fit(train, train_labels, epochs=50, batch_size=10,  verbose=2)
sio.savemate('history.mat', {'history': history})
scores = model.evaluate(test, test_labels)
print "Accuracy: %.2f" % (scores[1]*100.0)

model.save_weights('run1.h5')
