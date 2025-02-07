from keras.models import Model
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D, GlobalAveragePooling3D, Dense, Flatten, Dropout, Activation, SpatialDropout3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
import numpy as np


def model3d(vsize, sz=48, alpha=1.5, do_features=False):
    inputs = Input(vsize + (1,))
    
    def conv3dparams(**replace_params):
        params = { 'activation':ELU(), 'border_mode':'valid', 'init': 'he_normal' }
        params.update(replace_params)
        return params

    x = Convolution3D(sz, 3, 3, 3, name='b1c1', **conv3dparams())(inputs)
    x = BatchNormalization(name='b1c1_bn')(x)
    x = Convolution3D(sz, 1, 1, 1, name='b1c2', **conv3dparams())(x)
    x = BatchNormalization(name='b1c2_bn')(x)

    sz = int(sz * alpha)
    if vsize == (32,32,32):
        x = Convolution3D(sz, 3, 3, 3, name='b2c1_32', subsample=(2,2,2), **conv3dparams())(x)
    else:
        x = Convolution3D(sz, 3, 3, 3, name='b2c1', **conv3dparams())(x)
    x = BatchNormalization(name='b2c1_bn')(x)
    x = Convolution3D(sz, 1, 1, 1, name='b2c2', **conv3dparams())(x)
    x = BatchNormalization(name='b2c2_bn')(x)
    x = Convolution3D(sz, 3, 3, 3, name='b2c3', **conv3dparams())(x)
    x = BatchNormalization(name='b2c3_bn')(x)
    if vsize == (32,32,32):
        x = Convolution3D(sz, 3, 3, 3, name='b2c4_32', **conv3dparams())(x)
    else:
        x = Convolution3D(sz, 1, 1, 1, name='b2c4', **conv3dparams())(x)
    x = BatchNormalization(name='b2c4_bn')(x)
    x = SpatialDropout3D(0.2)(x)

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, name='b3c1', **conv3dparams())(x)
    x = BatchNormalization(name='b3c1_bn')(x)
    x = Convolution3D(sz, 1, 1, 1, name='b3c2', **conv3dparams())(x)
    x = BatchNormalization(name='b3c2_bn')(x)
    x = Convolution3D(sz, 3, 3, 3, name='b3c3', **conv3dparams())(x)
    x = BatchNormalization(name='b3c3_bn')(x)
    x = Convolution3D(sz, 1, 1, 1, name='b3c4', **conv3dparams())(x)
    x = BatchNormalization(name='b3c4_bn')(x)
    x = SpatialDropout3D(0.2)(x)

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, name='b4c1', **conv3dparams())(x)
    x = BatchNormalization(name='b4c1_bn')(x)
    x = Convolution3D(sz, 1, 1, 1, name='b4c2', **conv3dparams())(x)
    x = BatchNormalization(name='b4c2_bn')(x)
    x = Convolution3D(sz, 3, 3, 3, name='b4c3', **conv3dparams())(x)
    x = BatchNormalization(name='b4c3_bn')(x)
    x = Convolution3D(sz, 1, 1, 1, name='b4c4', **conv3dparams())(x)
    x = BatchNormalization(name='b4c4_bn')(x)
    x = SpatialDropout3D(0.5)(x)

    sz = int(sz * alpha)
    x = Convolution3D(sz, 2, 2, 2, name='b5c1', **conv3dparams())(x)
    x = BatchNormalization(name='b5c1_bn')(x)
    x = Convolution3D(sz, 1, 1, 1, name='b5c2', **conv3dparams())(x)
    x = BatchNormalization(name='b5c2_bn')(x)
    x = Convolution3D(sz, 1, 1, 1, name='b5c3', **conv3dparams())(x)
    x = BatchNormalization(name='b5c3_bn')(x)
    x = Convolution3D(1, 1, 1, 1, name='b5c4', **conv3dparams(activation='linear', border_mode='same'))(x)
    if not do_features:
        x = Flatten()(x)
        x = Activation('sigmoid')(x)

    model = Model(input=inputs, output=x)

    return model


def tiled_predict(model, image):
    s = 49
    d = 64
    m = 8
    full_result = np.zeros((image.shape[0]+d, image.shape[1]+d, image.shape[2]+d), dtype=np.float32)
    for i in range(0, int(np.ceil(image.shape[0]/s))):
        for j in range(0, int(np.ceil(image.shape[1]/s))):
            for k in range(0, int(np.ceil(image.shape[2]/s))):
                input_ = image[i*s:i*s+d,j*s:j*s+d,k*s:k*s+d]
                if input_.shape != (d,d,d):
                    input_ = np.pad(input_, ((0, d-input_.shape[0]), (0, d-input_.shape[1]), (0, d-input_.shape[2])), 'constant')
                result = model.predict(input_.reshape((1,d,d,d,1)), batch_size=1)
                full_result[i*s+m:(i+1)*s+m, j*s+m:(j+1)*s+m, k*s+m:(k+1)*s+m] = result[0,:,:,:,0]
    return full_result[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]]


def softmax_activations(x):
    r = x.reshape((-1,2))
    r_max = np.max(r, axis=-1)
    r_max = r_max[...,None]
    r_sum = np.sum(np.exp(r - r_max), axis=-1)
    r_sum = r_sum[...,None]
    p = np.exp(r - r_max) / r_sum
    p = p.reshape(x.shape)
    return p


def sigmoid_activations(x):  
    return np.exp(-np.logaddexp(0, -x))
