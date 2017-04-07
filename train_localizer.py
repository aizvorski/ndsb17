import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam, RMSprop

import data
import datagen
import net

import random
import skimage.transform
import json

import sys
import importlib
import datetime
import subprocess

SNAP_PATH = '/mnt/data/snap/'

config_name = sys.argv[1]
config = importlib.import_module(config_name)

fold = int(sys.argv[2])

weights_file = sys.argv[3]

run_id = 'localizer' + '__' + config_name + '__' + datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
print(run_id)

vsize = np.asarray([32,32,32])

df_nodes = data.ndsb17_get_df_nodes() 
df_nodes = df_nodes[(df_nodes["diameter_mm"]>=9)]

patient_ids = data.ndsb17_get_patient_ids_noncancer()
for k in range(fold):
    np.random.shuffle(patient_ids)

X_nodules, diams = data.ndsb17_get_all_nodules(np.asarray([64,64,64]), df_nodes)
#X_nodules = [x for x in X_nodules if x.shape == (64,64,64)] # FIXME this was a critical bug, nodules is filtered but diams is not
print("nodules", len(X_nodules))

X_nodules_train, X_nodules_test = data.kfold_split(X_cancer_nodules, fold)
diams_train, diams_test = data.kfold_split(diams, fold)

gen = datagen.batch_generator(vsize, patient_ids, X_nodules_train, diams_train)


def random_volume(image, vsize):
    pos = np.asarray([ np.random.randint(k, image.shape[k] - vsize[k]) for k in range(3) ])
    volume = image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
    return volume

# FIXME pass nodules split as input
# FIXME crop because expanded margin for rotation
test_nodules = np.stack(X_nodules_test)[:,16:16+32,16:16+32,16:16+32,None]
test_nodules = datagen.preprocess(test_nodules)
test_nodules = skimage.transform.downscale_local_mean(test_nodules, (1,2,2,2,1), clip=False)


num_test_volumes = 50
def get_test_volumes():
    test_volumes = []

    vsize = np.asarray([128,128,128])

    while len(test_volumes) < num_test_volumes:
        pid = np.random.choice(patient_ids)
        image = data.ndsb17_get_image(pid)
        segmented_image = data.ndsb17_get_segmented_image(pid)
        pos = np.asarray([ np.random.randint(k, image.shape[k] - vsize[k]) for k in range(3) ])
        segmented_volume = segmented_image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
        if np.count_nonzero(segmented_volume) == 0:
            continue
        volume = image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]

        test_volumes.append(volume)

    test_volumes = np.stack(test_volumes)[...,None]
    test_volumes = datagen.preprocess(test_volumes)
    test_volumes = skimage.transform.downscale_local_mean(test_volumes, (1,2,2,2,1), clip=False)

    return test_volumes


test_volumes = get_test_volumes()


def eval_model(model, volume_model):
    p_list = model.predict(test_nodules)[:,0]
    p_threshold = np.mean(sorted(p_list)[10:15]) # FIXME depends on size of X_nodules and tpr target
    print([ '%.4f' %(x) for x in sorted(p_list)[:10] ])
    #p_threshold = 0.99
    model.save_weights(SNAP_PATH + run_id + '.tmp.h5')
    volume_model.load_weights(SNAP_PATH + run_id + '.tmp.h5')

    fpr_list = []
    fpr90_list = []
    for n in range(num_test_volumes):
        test_result = volume_model.predict(test_volumes[n:n+1], batch_size=1)[:,:,:,0]
        test_p = net.sigmoid_activations(test_result)
        fpr = np.count_nonzero(test_p[0,:,:,:] > p_threshold) / test_volumes[n].size
        fpr_list.append(fpr)
        fpr90 = np.count_nonzero(test_p[0,:,:,:] > 0.880797) / test_volumes[n].size
        fpr90_list.append(fpr90)
    
    return np.mean(fpr_list), np.mean(fpr90_list), p_threshold, fpr_list, fpr90_list, p_list


history = {'loss':[], 'acc':[], 'fpr':[], 'p_threshold':[], 'p_list':[]}
history['version'] = subprocess.check_output('git describe --always --dirty', shell=True).decode('ascii').strip()
history['argv'] = sys.argv

model = net.model3d((16, 16, 16), sz=config.feature_sz, alpha=config.feature_alpha)
print(model.summary())
volume_model = net.model3d((64, 64, 64), sz=config.feature_sz, alpha=config.feature_alpha, do_features=True)

def get_optimizer(lr):
    if config.optimizer == 'rmsprop':
        optimizer = RMSprop(lr=lr)
    elif config.optimizer == 'adam':
        optimizer = Adam(lr=lr)
    elif config.optimizer == 'nadam':
        optimizer = Nadam(lr=lr)
    elif config.optimizer == 'sgd':
        optimizer = SGD(lr=lr, momentum=0.9, nesterov=True)
    return optimizer

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=get_optimizer(config.lr))
model.load_weights(SNAP_PATH + 'localizer.h5')

fom_best = -1e+6
for e in range(1, config.num_epochs):
    h = model.fit_generator(
        gen,
        config.samples_per_epoch,
        nb_epoch=1,
        verbose=1)

    fpr, fpr90, p_threshold, fpr_list, fpr90_list, p_list = eval_model(model, volume_model)
    print("fpr", fpr, "fpr90", fpr90, "std", np.std(fpr_list), "p_threshold", p_threshold)
    history['loss'].append(h.history['loss'][0])
    history['acc'].append(h.history['acc'][0])
    history['fpr'].append(fpr)
    history['p_threshold'].append(float(p_threshold))
    history['p_list'].append([ float(x) for x in p_list])

    model.save_weights(SNAP_PATH + run_id + '.{:04d}'.format(e) + '.h5')

    with open(SNAP_PATH + run_id + '.log.json', 'w') as fh:
        json.dump(history, fh)

    # trade off 1e-6 fpr versus 0.1 tpr
    fom = np.mean(p_list) - 0.1 * fpr90 / 1e-6
    print("fom", fom)
    if fom > fom_best:
        fom_best = fom
        print("*** saving best result")
        model.save_weights(SNAP_PATH + weights_file)

    if e == config.lr_step_num_epochs:
        print("*** reloading from best result")

        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=get_optimizer(config.lr * config.lr_step_multiplier))
        model.load_weights(SNAP_PATH + weights_file)
