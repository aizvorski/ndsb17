from keras.models import Model
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D, GlobalAveragePooling3D, Dense, Flatten, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
import numpy as np


def model3d(vsize, sz=48, alpha=1.5, do_features=False):
    inputs = Input(vsize + (1,))
    
    def conv3dparams(**replace_params):
        params = { 'activation':'linear', 'border_mode':'valid', 'init': 'he_normal' }
        params.update(replace_params)
        return params

    x = Convolution3D(sz, 3, 3, 3, **conv3dparams())(inputs)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(sz, 1, 1, 1, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(sz, 1, 1, 1, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(sz, 1, 1, 1, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(sz, 1, 1, 1, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(sz, 1, 1, 1, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(sz, 1, 1, 1, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(sz, 1, 1, 1, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))

    sz = int(sz * alpha)
    x = Convolution3D(sz, 2, 2, 2, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(sz, 1, 1, 1, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(sz, 1, 1, 1, **conv3dparams())(x)
    x = ELU()(BatchNormalization()(x))
    x = Convolution3D(2, 1, 1, 1, **conv3dparams(activation='linear', border_mode='same'))(x)
    if not do_features:
        x = Flatten()(x)
        x = Activation('softmax')(x)

    model = Model(input=inputs, output=x)

    return model


def tiled_predict(model, image):
    s = 49
    d = 64
    m = 8
    full_result = np.zeros((image.shape[0]+d, image.shape[1]+d, image.shape[2]+d, 2), dtype=np.float32)
    full_result[:,:,:,0] = 1e+6
    for i in range(0, int(np.ceil(image.shape[0]/s))):
        for j in range(0, int(np.ceil(image.shape[1]/s))):
            for k in range(0, int(np.ceil(image.shape[2]/s))):
                input_ = image[i*s:i*s+d,j*s:j*s+d,k*s:k*s+d]
                if input_.shape != (d,d,d):
                    input_ = np.pad(input_, ((0, d-input_.shape[0]), (0, d-input_.shape[1]), (0, d-input_.shape[2])), 'constant')
                result = model.predict(input_.reshape((1,d,d,d,1)), batch_size=1)
                full_result[i*s+m:(i+1)*s+m, j*s+m:(j+1)*s+m, k*s+m:(k+1)*s+m,:] = result
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
