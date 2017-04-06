from keras.models import Model
from keras.layers import Input, merge, Convolution3D, GlobalMaxPooling3D, Dense, Flatten, Dropout, Activation, SpatialDropout3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
import numpy as np


def model3d_layers(sz=48, alpha=1.5, do_features=False):

    layers = []
    
    def conv3dparams(**replace_params):
        params = { 'activation':ELU(), 'border_mode':'valid', 'init': 'he_normal' }
        params.update(replace_params)
        return params

    layers.append( Convolution3D(sz, 3, 3, 3, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( Convolution3D(sz, 1, 1, 1, **conv3dparams()) )
    layers.append( "BatchNormalization" )

    sz = int(sz * alpha)
    # if vsize == (32,32,32):
    #     layers.append( Convolution3D(sz, 3, 3, 3, subsample=(2,2,2), **conv3dparams()) )
    # else:
    layers.append( Convolution3D(sz, 3, 3, 3, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( Convolution3D(sz, 1, 1, 1, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( Convolution3D(sz, 3, 3, 3, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    # if vsize == (32,32,32):
    #     layers.append( Convolution3D(sz, 3, 3, 3, **conv3dparams()) )
    # else:
    layers.append( Convolution3D(sz, 1, 1, 1, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( SpatialDropout3D(0.2) )

    sz = int(sz * alpha)
    layers.append( Convolution3D(sz, 3, 3, 3, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( Convolution3D(sz, 1, 1, 1, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( Convolution3D(sz, 3, 3, 3, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( Convolution3D(sz, 1, 1, 1, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( SpatialDropout3D(0.2) )

    sz = int(sz * alpha)
    layers.append( Convolution3D(sz, 3, 3, 3, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( Convolution3D(sz, 1, 1, 1, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( Convolution3D(sz, 3, 3, 3, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( Convolution3D(sz, 1, 1, 1, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( SpatialDropout3D(0.5) )

    sz = int(sz * alpha)
    layers.append( Convolution3D(sz, 2, 2, 2, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( Convolution3D(sz, 1, 1, 1, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( Convolution3D(sz, 1, 1, 1, **conv3dparams()) )
    layers.append( "BatchNormalization" )
    layers.append( Convolution3D(1, 1, 1, 1, **conv3dparams(activation='linear', border_mode='same')) )

    layers.append( GlobalMaxPooling3D() )
    layers.append( Activation('sigmoid') )


    return layers


def model3d_build(vsize, layers):
    inputs = Input(vsize + (1,))
    x = inputs

    for f in layers:
        if f == "BatchNormalization":
            #continue
            f = BatchNormalization()
        x = f(x)

    model = Model(input=inputs, output=x)
    return model
