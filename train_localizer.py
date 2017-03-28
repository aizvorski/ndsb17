import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam, RMSprop

import data
import datagen
import net

import random
import scipy.ndimage.interpolation
import json

import sys
import importlib
import datetime
import subprocess

config_name = sys.argv[1]
config = importlib.import_module(config_name)

run_id = config_name + '__' + datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
print(run_id)

SNAP_PATH = '/mnt/data/snap/'

vsize = np.asarray([32,32,32])

df_nodes = data.ndsb17_get_df_nodes() 
df_nodes = df_nodes[(df_nodes["diameter_mm"]>10)]

patient_ids = data.ndsb17_get_patient_ids_noncancer()

X_nodules, diams = data.ndsb17_get_all_nodules(vsize, df_nodes)

gen = datagen.batch_generator(vsize, patient_ids, X_nodules[:-50], diams[:-50])


def random_volume(image, vsize):
    pos = np.asarray([ np.random.randint(k, image.shape[k] - vsize[k]) for k in range(3) ])
    volume = image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
    return volume

patient_ids_noncancer = data.ndsb17_get_patient_ids_noncancer()

test_nodules = np.stack(X_nodules[-50:])[...,None] # FIXME pass nodules as input
test_nodules = datagen.preprocess(test_nodules)
test_nodules = scipy.ndimage.interpolation.zoom(test_nodules, (1, 0.5, 0.5, 0.5, 1), order=1)

test_volumes = []

for n in range(10):
    pid = random.choice(patient_ids_noncancer)
    image = data.ndsb17_get_image(pid)
    # info = data.ndsb17_get_info(pid)
    test_volume = random_volume(image, (128,128,128))
    test_volume = datagen.preprocess(test_volume)
    test_volume = scipy.ndimage.interpolation.zoom(test_volume, (0.5, 0.5, 0.5), order=1)
    test_volumes.append(test_volume)

test_volumes = np.stack(test_volumes)[...,None]

def eval_model(model, volume_model, num_evals=10):
    p_list = model.predict(test_nodules)[:,1]
    p_threshold = np.mean(sorted(p_list)[10:15]) # FIXME depends on size of X_nodules and tpr target
    print([ '%.4f' %(x) for x in sorted(p_list)[:10] ])
    #p_threshold = 0.99
    model.save_weights(SNAP_PATH + run_id + '.tmp.h5')
    volume_model.load_weights(SNAP_PATH + run_id + '.tmp.h5')

    fpr_list = []
    for n in range(num_evals):
        test_result = volume_model.predict(test_volumes[n:n+1], batch_size=1)
        test_p = net.softmax_activations(test_result)
        fpr = np.count_nonzero(test_p[0,:,:,:,1] > p_threshold) / test_volume.size
        fpr_list.append(fpr)
    
    return np.mean(fpr_list), p_threshold, fpr_list, p_list

history = {'loss':[], 'acc':[], 'fpr':[], 'p_threshold':[], 'p_list':[]}
history['version'] = subprocess.check_output('git describe --always --dirty', shell=True).decode('ascii').strip()
history['argv'] = sys.argv

model = net.model3d((16, 16, 16), sz=config.feature_sz, alpha=config.feature_alpha)
print(model.summary())
volume_model = net.model3d((64, 64, 64), sz=config.feature_sz, alpha=config.feature_alpha, do_features=True)

optimizer = RMSprop(lr=config.lr)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)


for e in range(config.num_epochs):
    h = model.fit_generator(
        gen,
        config.samples_per_epoch,
        nb_epoch=1,
        verbose=1)

    fpr, p_threshold, fpr_list, p_list = eval_model(model, volume_model)
    print("fpr", fpr, "std", np.std(fpr_list), "p_threshold", p_threshold)
    history['loss'].append(h.history['loss'][0])
    history['acc'].append(h.history['acc'][0])
    history['fpr'].append(fpr)
    history['p_threshold'].append(float(p_threshold))
    history['p_list'].append([ float(x) for x in p_list])

    model.save_weights(SNAP_PATH + run_id + '.{:04d}'.format(e) + '.h5')

    with open(SNAP_PATH + run_id + '.log.json', 'w') as fh:
        json.dump(history, fh)
