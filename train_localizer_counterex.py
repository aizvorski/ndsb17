import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam, RMSprop

import data
import datagen
import net_dual

import predict_localizer
gpu_id = 0 # HACK for running predict_localizer without multi-gpu

import random
import skimage.transform
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

vsize_x2 = np.asarray([32,32,32])
vsize = np.asarray([16,16,16])

df_nodes = data.ndsb17_get_df_nodes() 
df_nodes = df_nodes[(df_nodes["diameter_mm"]>10)]

patient_ids = data.ndsb17_get_patient_ids_noncancer()

X_nodules, diams = data.ndsb17_get_all_nodules(np.asarray([64,64,64]), df_nodes)
print("nodules", len(X_nodules))

gen = datagen.batch_generator(vsize_x2, patient_ids, X_nodules[:-50], diams[:-50], batch_size=64)

# FIXME pass nodules split as input
# FIXME crop because expanded margin for rotation
# test_nodules = np.stack(X_nodules[-50:])[:,16:16+32,16:16+32,16:16+32,None]
# test_nodules = datagen.preprocess(test_nodules)
# test_nodules = skimage.transform.downscale_local_mean(test_nodules, (1,2,2,2,1), clip=False)

# test_volumes = []

# for n in range(10):
#     pid = random.choice(patient_ids_noncancer)
#     image = data.ndsb17_get_image(pid)
#     # info = data.ndsb17_get_info(pid)
#     test_volume = random_volume(image, (128,128,128))
#     test_volume = datagen.preprocess(test_volume)
#     test_volume = skimage.transform.downscale_local_mean(test_volume, (1,2,2,2,1), clip=False)
#     test_volumes.append(test_volume)

# test_volumes = np.stack(test_volumes)[...,None]

# def eval_model(model, volume_model, num_evals=10):
#     p_list = model.predict(test_nodules)[:,1]
#     p_threshold = np.mean(sorted(p_list)[10:15]) # FIXME depends on size of X_nodules and tpr target
#     print([ '%.4f' %(x) for x in sorted(p_list)[:10] ])
#     #p_threshold = 0.99
#     model.save_weights(SNAP_PATH + run_id + '.tmp.h5')
#     volume_model.load_weights(SNAP_PATH + run_id + '.tmp.h5')

#     fpr_list = []
#     for n in range(num_evals):
#         test_result = volume_model.predict(test_volumes[n:n+1], batch_size=1)
#         test_p = net.softmax_activations(test_result)
#         fpr = np.count_nonzero(test_p[0,:,:,:,1] > p_threshold) / test_volume.size
#         fpr_list.append(fpr)
    
#     return np.mean(fpr_list), p_threshold, fpr_list, p_list

# history = {'loss':[], 'acc':[], 'fpr':[], 'p_threshold':[], 'p_list':[]}
# history['version'] = subprocess.check_output('git describe --always --dirty', shell=True).decode('ascii').strip()
# history['argv'] = sys.argv

layers = net_dual.model3d_layers(sz=32, alpha=1.5)

model = net_dual.model3d_build((16, 16, 16), layers)
print(model.summary())

volume_model = net_dual.model3d_build((64, 64, 64), layers)
print(volume_model.summary())

predict_localizer.volume_model = volume_model

# if config.optimizer == 'rmsprop':
#     optimizer = RMSprop(lr=config.lr)
# elif config.optimizer == 'adam':
#     optimizer = Adam(lr=config.lr)
# elif config.optimizer == 'nadam':
#     optimizer = Nadam(lr=config.lr)
# elif config.optimizer == 'sgd':
#     optimizer = SGD(lr=config.lr, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=0.001))
#volume_model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=0.001))

for e in range(10000):
    h = model.fit_generator(
        gen,
        1000,
        nb_epoch=1,
        verbose=1)

    model.save_weights(SNAP_PATH + run_id + '.tmp.h5')
    volume_model.load_weights(SNAP_PATH + run_id + '.tmp.h5')

    pid = np.random.choice(patient_ids)
    print(pid)
    predicted_image, labels = predict_localizer.predict_localizer(pid)
    label_boxes, label_sizes, label_activities_sum, label_activities_max = labels

    image = data.ndsb17_get_image(pid).astype(np.float32)
    image_2mm = skimage.transform.downscale_local_mean(image, (2,2,2), clip=False)
    
    X = []
    for pos in np.argwhere(predicted_image > 2):
        pos -= 8
        volume = image_2mm[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
        if volume.shape != list(vsize):
            continue
        X.append( volume )

    if len(X) == 0:
        continue
    if len(X) > 1000:
        X = np.random.choice(X, size=1000, replace=False)
    X = np.stack(X)[...,None]
    X = datagen.preprocess(X)
    y = np.zeros((X.shape[0]))

    print(X.shape)

    h = model.fit(
        X, y,
        batch_size=64,
        nb_epoch=1,
        verbose=1)


    # fpr, p_threshold, fpr_list, p_list = eval_model(model, volume_model)
    # print("fpr", fpr, "std", np.std(fpr_list), "p_threshold", p_threshold)
    # history['loss'].append(h.history['loss'][0])
    # history['acc'].append(h.history['acc'][0])
    # history['fpr'].append(fpr)
    # history['p_threshold'].append(float(p_threshold))
    # history['p_list'].append([ float(x) for x in p_list])

    model.save_weights(SNAP_PATH + run_id + '.{:04d}'.format(e) + '.h5')

    # with open(SNAP_PATH + run_id + '.log.json', 'w') as fh:
    #     json.dump(history, fh)
