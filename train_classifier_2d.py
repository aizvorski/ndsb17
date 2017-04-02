import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Flatten

import data
import datagen
#import net
import xception

import random
import scipy.ndimage.interpolation
import json
import pickle

import sys
import importlib
import datetime
import subprocess

config_name = sys.argv[1]
config = importlib.import_module(config_name)

run_id = config_name + '__' + datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
print(run_id)

SNAP_PATH = '/mnt/data/snap/'

vsize = np.asarray([64,64,64])

df_nodes = data.ndsb17_get_df_nodes() 
df_nodes = df_nodes[(df_nodes["diameter_mm"]>10)]

patient_ids = data.ndsb17_get_patient_ids_noncancer()

X_cancer_nodules, cancer_diams = data.ndsb17_get_all_nodules(np.asarray([64,64,64]), df_nodes)
#X_cancer_nodules = [x for x in X_cancer_nodules if x.shape == (64,64,64)] # FIXME not all results are full size
print("cancer nodules", len(X_cancer_nodules))


X_benign_nodules, benign_diams = data.ndsb17_get_predicted_nodules(np.asarray([64,64,64]), patient_ids)
benign_diams = [64 for x in benign_diams]


def volume_3d_to_2d(X):
    tmp = X[:,X.shape[1]//2,:,:,0]
    X = np.stack((tmp, tmp, tmp), axis=-1)
    return X

def batch_generator_3d_to_2d(gen):
    while True:
        X, y = next(gen)
        tmp = X[:,X.shape[1]//2,:,:,0]
        X = np.stack((tmp, tmp, tmp), axis=-1)
        yield X, y

gen = datagen.batch_generator_ab(vsize, patient_ids, X_benign_nodules[:-50], benign_diams[:-50], X_cancer_nodules[:-50], cancer_diams[:-50], do_downsample=False)

gen_2d = batch_generator_3d_to_2d(gen)

test_nodules = np.stack(X_benign_nodules[-50:] + X_cancer_nodules[-50:])[:,:,:,:,None]
test_nodules = datagen.preprocess(test_nodules)
test_nodules_2d = volume_3d_to_2d(test_nodules)
#test_nodules = scipy.ndimage.interpolation.zoom(test_nodules, (1, 0.5, 0.5, 0.5, 1), order=1)
test_y = np.zeros((test_nodules.shape[0], 2), dtype=np.int)
test_y[:50,0] = 1
test_y[50:,1] = 1


history = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
history['version'] = subprocess.check_output('git describe --always --dirty', shell=True).decode('ascii').strip()
history['argv'] = sys.argv

model1 = xception.Xception(input_shape=(64,64,3), include_top=False)

classes = 2
x = GlobalAveragePooling2D(name='avg_pool')( model1.outputs[0] )
x = Dense(classes, activation='softmax', name='predictions')(x)

model = Model(model1.inputs, x, name='xception')
print(model.summary())

if config.optimizer == 'rmsprop':
    optimizer = RMSprop(lr=config.lr)
elif config.optimizer == 'adam':
    optimizer = Adam(lr=config.lr)
elif config.optimizer == 'nadam':
    optimizer = Nadam(lr=config.lr)
elif config.optimizer == 'sgd':
    optimizer = SGD(lr=config.lr, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
# model.load_weights('/mnt/data/snap/config_baseline2__20170329095350.0050.h5')

for e in range(config.num_epochs):
    h = model.fit_generator(
        gen_2d,
        config.samples_per_epoch,
        nb_epoch=1,
        verbose=1,
        validation_data=(test_nodules_2d, test_y))

    print(h.history)
    history['loss'].append(h.history['loss'][0])
    history['acc'].append(h.history['acc'][0])

    model.save_weights(SNAP_PATH + run_id + '.{:04d}'.format(e) + '.h5')

    with open(SNAP_PATH + run_id + '.log.json', 'w') as fh:
        json.dump(history, fh)
