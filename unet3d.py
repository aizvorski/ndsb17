from keras.models import Model
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D

w, h, d = 64,64,64

def get_unet3d():
    inputs = Input((w, h, d, 1))
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv4], mode='concat', concat_axis=4)
    conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3], mode='concat', concat_axis=4)
    conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2], mode='concat', concat_axis=4)
    conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling3D(size=(2, 2, 2))(conv8), conv1], mode='concat', concat_axis=4)
    conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

model = get_unet3d()
print model.summary()