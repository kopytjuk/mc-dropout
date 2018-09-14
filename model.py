import numpy as np
import matplotlib.pyplot as  plt
import tensorflow as tf

from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from sklearn.utils import shuffle

rate = 0.2
structure = [20, 10]
N = 30000
data_split = 0.7
val_split = 0.3
patience = 1000
epochs = 15000
opt = Adam(lr=0.01)
mc_samples = 1000

#X = np.linspace(-1,1, num = N).reshape((-1,1))
X = np.random.uniform(low=-1.0, high=0.5, size = (N,1))
Y = np.sin(5*X)

X_train = X[:int(N*data_split),:]
Y_train = Y[:int(N*data_split),:]

X_test = X[int(N*data_split):,:]
Y_test = Y[int(N*data_split):,:]

assert len(Y.shape) == 2

with tf.device("/gpu:0"):
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape = [1]))
    model.add(Dropout(rate = rate))
    #model.add(Dense(20, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(rate = rate))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(rate = rate))
    model.add(Dense(1))

model.compile(optimizer = opt, loss = 'mse')

acquisition_func = K.function([model.input, K.learning_phase()], [model.output])

early_callback = EarlyStopping(monitor='val_loss', min_delta=0,patience=patience)
checkpoint = ModelCheckpoint(filepath = 'models/{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only = True)
#checkpoint = ModelCheckpoint(filepath = 'models/model.hdf5', save_best_only = True)

callbacks = [early_callback, checkpoint]
callbacks = [early_callback]

#X_train_shuffled, Y_train_shuffled = shuffle((X_train, Y_train))

model.fit(X, Y, batch_size=10000, epochs=epochs, verbose=1, callbacks=callbacks, shuffle=True,
          validation_split=val_split)

#model = load_model('net.hdf5')

X_test_2 = np.random.uniform(low=-1.0, high=1.0, size = (100,1))

T_in = np.vstack([X_test_2]*mc_samples)

# get output
T_out = acquisition_func([T_in, 1])[0]

T_in = T_in.reshape((-1,))
T_out = T_out.reshape((-1,))
plt.scatter(T_in, T_out, s=0.1)
plt.scatter(X,Y)
plt.savefig('test.png')