import numpy as np
import numpy.random
import skimage
import skimage.transform
import scipy.misc
import joblib

batch_size = 32

numpy.random.seed(1234)


def load_data():
    diams = np.load("luna16_cube64/nodules/all_diams.npy")
    N = diams.shape[0]
    
    stack4d = np.zeros((N,64,64,64),dtype=np.float32)

    for k in range(N):
        img_node = np.load("luna16_cube64/nodules/" + str(k) + ".npy")
        if img_node.shape != (64,64,64):
            print "bad shape " + str(img_node.shape)
            continue
        stack4d[k] = img_node

    X = stack4d
    return X, diams

X, diams = load_data()

X = (X + 571.28314) / 425.15402

def image_generator(X, diams, do_augment=False, do_easy=False):
    while True:
        N = X.shape[0]
        idx = np.random.permutation(N)
        X2, diams2 = X[idx], diams[idx]

        for i in range(X2.shape[0] // batch_size - 1):
            X_ = np.zeros((batch_size, 32, 32, 1), dtype=np.float32)
            Y_ = np.zeros((batch_size, 2), dtype=np.float32)

            for k in range(batch_size):
                j = i*batch_size + k
                # sample from center or edge
                c = np.random.choice( (True, False) )

                r = int(round(diams2[j]*0.5))
                if r > 31:
                    r = 31

                if c:
                    if do_easy:
                        z = 32
                    else:
                        z = np.random.randint(32-r, 32+r)
                else:
                    if do_easy:
                        z = np.random.choice( (0, 63) )
                    else:
                        z = np.random.choice( (np.random.randint(0, 32-r), np.random.randint(32+r, 64)) )

                # print c, r, z
                
                img = X2[j, z]
                if do_augment:
                    img2 = skimage.transform.rotate(img, np.random.random()*360, resize=False, preserve_range=True)
                    x0, y0 = np.random.randint(16-5, 16+5+1), np.random.randint(16-5, 16+5+1)
                else:
                    img2 = img
                    x0, y0 = 16,16
                X_[k,:,:,0] = img2[x0:x0+32, y0:y0+32]
                
                # make one-hot encoding
                if c:
                    Y_[k, 1] = 1
                else:
                    Y_[k, 0] = 1

            yield X_, Y_

import os.path
with open(os.path.expanduser("~/.keras/keras.json"), "w") as fh:
    fh.write('{ "backend": "tensorflow", "image_dim_ordering": "tf", "epsilon": 1e-07, "floatx": "float32" }')
    #fh.write('{ "backend": "theano", "image_dim_ordering": "th", "epsilon": 1e-07, "floatx": "float32" }')


################################################################################

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import sklearn.cross_validation

# from allcnn32 import make_model

# from cifar10_tiny import make_model
# model = make_model()

# from xception import Xception
# model = Xception(weights=None)

from vgg16 import VGG16
model = VGG16(weights=None)

# sgd = SGD(lr=0.0003, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print model.summary()



snapname = '/mnt/data/snap/junk'
joblib.dump(model.to_json(), snapname + '.model')
checkpointer = ModelCheckpoint(filepath=snapname + '.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1)

kf = sklearn.cross_validation.KFold(X.shape[0], n_folds=10)
for t, v in kf:
    break
X_train, X_valid = X[t], X[v]
diams_train, diams_valid = diams[t], diams[v]

samples_per_epoch = (X_train.shape[0]//batch_size)*batch_size
nb_val_samples = (X_valid.shape[0]//batch_size)*batch_size

h = model.fit_generator(image_generator(X_train, diams_train), 
                samples_per_epoch=samples_per_epoch, 
                nb_epoch=100, 
                # metrics=['accuracy'],
                # callbacks=[checkpointer],
                validation_data=image_generator(X_valid, diams_valid, do_augment=False, do_easy=True), # change this to False
                nb_val_samples=nb_val_samples)
