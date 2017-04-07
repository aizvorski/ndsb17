import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam, RMSprop

import data
import datagen
import net

import random
import skimage.transform
import json
import pickle

import sys
import importlib
import datetime
import subprocess

SNAP_PATH = '/mnt/data/snap/'

config_name = sys.argv[1]
config = importlib.import_module(config_name)

fold = int(sys.argv[2])

localizer_output_dir = sys.argv[3]

weights_file = sys.argv[4]

run_id = 'classifier' + '__' + config_name + '__' + datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
print(run_id)


vsize = np.asarray([32,32,32])

df_nodes = data.ndsb17_get_df_nodes() 
df_nodes = df_nodes[(df_nodes["diameter_mm"]>10)]

patient_ids = data.ndsb17_get_patient_ids_noncancer()

X_cancer_nodules, cancer_diams = data.ndsb17_get_all_nodules(np.asarray([64,64,64]), df_nodes)
print("cancer nodules", len(X_cancer_nodules))

X_localizer_nodules = data.ndsb17_get_predicted_nodules(np.asarray([64,64,64]), patient_ids, SNAP_PATH+localizer_output_dir)
print("localizer nodules", len(X_localizer_nodules))

def batch_generator_ab(vsize, X_nodules_a, X_nodules_b, batch_size=64, do_downscale=True):
    while True:
        X = np.zeros((batch_size,) + vsize + (1,), dtype=np.float32)
        y = np.zeros((batch_size), dtype=np.int)
        n = 0
        while n < batch_size:
            if np.random.choice([True, False]):
                volume = datagen.make_augmented(vsize, np.random.choice(X_nodules_a))
                y[n] = 0
            else:
                volume = datagen.make_augmented(vsize, np.random.choice(X_nodules_b))
                y[n] = 1
            X[n,:,:,:,0] = volume
            n += 1
        X = datagen.preprocess(X)
        if do_downscale:
            X = skimage.transform.downscale_local_mean(X, (1,2,2,2,1), clip=False)
        yield X, y


gen = batch_generator_ab(vsize, X_localizer_nodules[:-50], X_cancer_nodules[:-50])

test_nodules = np.stack(X_localizer_nodules[-50:] + X_cancer_nodules[-50:])[:,16:16+32,16:16+32,16:16+32,None]
test_nodules = datagen.preprocess(test_nodules)
test_nodules = skimage.transform.downscale_local_mean(test_nodules, (1,2,2,2,1), clip=False)
test_y = np.zeros((test_nodules.shape[0],), dtype=np.int)
test_y[50:] = 1

history = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
history['version'] = subprocess.check_output('git describe --always --dirty', shell=True).decode('ascii').strip()
history['argv'] = sys.argv

model = net.model3d((16, 16, 16), sz=config.feature_sz, alpha=config.feature_alpha)
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
model.load_weights('/mnt/data/snap/localizer_good.h5')

for e in range(config.num_epochs):
    h = model.fit_generator(
        gen,
        config.samples_per_epoch,
        nb_epoch=1,
        verbose=1,
        validation_data=(test_nodules, test_y))

    print(h.history)
    history['loss'].append(h.history['loss'][0])
    history['acc'].append(h.history['acc'][0])
    history['val_loss'].append(h.history['val_loss'][0])
    history['val_acc'].append(h.history['val_acc'][0])

    model.save_weights(SNAP_PATH + run_id + '.{:04d}'.format(e) + '.h5')

    with open(SNAP_PATH + run_id + '.log.json', 'w') as fh:
        json.dump(history, fh)
