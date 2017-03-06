import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D, GlobalAveragePooling3D, Dense
from keras.optimizers import SGD

w, h, d = 64,64,64

alpha = 2

def get_test3d():
    inputs = Input((w, h, d, 1))
    sz = 32

    x = Convolution3D(sz, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    x = Convolution3D(sz, 3, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution3D(sz, 3, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution3D(sz, 3, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution3D(sz, 3, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution3D(sz, 3, 3, 3, activation='relu', border_mode='same')(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(2, activation='sigmoid')(x)

    model = Model(input=inputs, output=x)

    return model

model = get_test3d()
print(model.summary())

sgd = SGD(lr=0.0003, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

batch_size=32

batch_num=100

model.fit(
    np.zeros((batch_size*batch_num, w, h, d, 1), dtype=np.float32), 
    np.zeros((batch_size*batch_num, 2), dtype=np.float32), 
    batch_size=batch_size,
    nb_epoch=10)

