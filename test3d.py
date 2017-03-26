import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D, GlobalAveragePooling3D, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

## data loading

# def load_data():
#     diams = np.load("luna16_cube64/nodules/all_diams.npy")
#     N = diams.shape[0]
    
#     stack4d = np.zeros((N,64,64,64),dtype=np.float32)

#     for k in range(N):
#         img_node = np.load("luna16_cube64/nodules/" + str(k) + ".npy")
#         if img_node.shape != (64,64,64):
#             print("bad shape " + str(img_node.shape))
#             continue
#         stack4d[k] = img_node

#     X = stack4d
#     return X, diams

# X, diams = load_data()

# X = (X + 571.28314) / 425.15402

# N = X.shape[0]
# X2 = np.concatenate( (X[:,16:16+32,16:16+32,16:16+32], X[:,0:0+32,0:0+32,0:0+32]), axis=0 )
# X2 = X2[..., np.newaxis]

# labels = np.zeros( (N*2, 2), dtype=np.int )
# labels[ 0:N ] = 1
# idx = np.random.permutation(N*2)
# X2, labels = X2[idx], labels[idx]

# print(X2.shape)
# print(labels.shape)

# #import sys
# #sys.exit()

# import sklearn.cross_validation

# kf = sklearn.cross_validation.KFold(X2.shape[0], n_folds=5)
# for t, v in kf:
#     break
# X_train, X_valid = X2[t], X2[v]
# labels_train, labels_valid = labels[t], labels[v]


w, h, d = 32, 32, 32

alpha = 2

def get_test3d():
    inputs = Input((w, h, d, 1))
    sz = 32

    conv3dparams = { 'activation':'relu', 'border_mode':'same' }

    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(inputs)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x) # 16x16

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x) # 8x8

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x) # 4x4

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x) # 2x2

    sz = int(sz * alpha)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = Convolution3D(sz, 3, 3, 3, **conv3dparams)(x)
    x = BatchNormalization()(x)

    x = Convolution3D(sz, 2, 2, 2, activation='relu', border_mode='valid')(x)
    x = Flatten()(x)
    # x = GlobalAveragePooling3D()(x)
    x = Dense(2, activation='sigmoid')(x)

    model = Model(input=inputs, output=x)

    return model

model = get_test3d()
print(model.summary())

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

batch_size=32

batch_num=100




model.fit(
    np.zeros((batch_size*batch_num, w, h, d, 1), dtype=np.float32), 
    np.zeros((batch_size*batch_num, 2), dtype=np.float32), 
    batch_size=batch_size,
    nb_epoch=10)

# model.fit(
#     X2, 
#     labels, 
#     batch_size=batch_size,
#     nb_epoch=10,
#     validation_split=0.2)

