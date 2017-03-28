import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D, GlobalAveragePooling3D, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD


w, h, d = 32, 32, 32

alpha = 2

def get_test3d():
    inputs = Input((w, h, d, 1))
    sz = 32

    conv3dparams = { 'activation':'relu', 'border_mode':'valid' }

    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(inputs)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = BatchNormalization()(x)
    # x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = BatchNormalization()(x)
    # x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = BatchNormalization()(x)
    # x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    sz = int(sz * alpha)
    # x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    # x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    # x = BatchNormalization()(x)

    x = Convolution3D(sz, 2, 2, 2, activation='relu', border_mode='valid')(x)
    x = Flatten()(x)
    # x = GlobalAveragePooling3D()(x)
    x = Dense(2, activation='sigmoid')(x)

    model = Model(input=inputs, output=x)

    return model

model = get_test3d()
print(model.summary())


import sys
sys.exit(0)



sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)



X = np.load("/mnt/data/luna16/sets/toy_set_v1_20k.npy")
y = np.zeros((20000,2))
y[:10000,0] = 1
y[10000:,1] = 1

idx = np.random.permutation(X.shape[0])
X, y = X[idx], y[idx]


batch_size=32

model.fit(
    X, 
    y, 
    batch_size=batch_size,
    nb_epoch=10,
    validation_split=0.1)
