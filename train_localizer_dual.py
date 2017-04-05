import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam, RMSprop

import data
import datagen
import net_dual

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

vsize = np.asarray([32,32,32])

df_nodes = data.ndsb17_get_df_nodes() 
df_nodes = df_nodes[(df_nodes["diameter_mm"]>10)]

patient_ids = data.ndsb17_get_patient_ids_noncancer()

X_nodules, diams = data.ndsb17_get_all_nodules(np.asarray([64,64,64]), df_nodes)
print("nodules", len(X_nodules))

gen = datagen.batch_generator(vsize, patient_ids, X_nodules[:-50], diams[:-50])

def random_volume(image, vsize):
    pos = np.asarray([ np.random.randint(k, image.shape[k] - vsize[k]) for k in range(3) ])
    volume = image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
    return volume

def volume_batch_generator(vsize, patient_ids, batch_size=64, do_downscale=True):
    while True:
        X = np.zeros((batch_size,) + vsize + (1,), dtype=np.float32)
        y = np.zeros((batch_size, 2), dtype=np.int)

        for n in range(batch_size):
            pid = np.random.choice(patient_ids)
            image = data.ndsb17_get_image(pid)
            volume = random_volume(image, vsize)
            X[n,:,:,:,0] = volume
            y[n,0] = 1

        if do_downscale:
            X = skimage.transform.downscale_local_mean(X, (1,2,2,2,1), clip=False)
        X = datagen.preprocess(X)
        yield X, y

volume_gen = volume_batch_generator((128,128,32), patient_ids, batch_size=8)


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

volume_model = net_dual.model3d_build((64, 64, 16), layers)
print(volume_model.summary())

# if config.optimizer == 'rmsprop':
#     optimizer = RMSprop(lr=config.lr)
# elif config.optimizer == 'adam':
#     optimizer = Adam(lr=config.lr)
# elif config.optimizer == 'nadam':
#     optimizer = Nadam(lr=config.lr)
# elif config.optimizer == 'sgd':
#     optimizer = SGD(lr=config.lr, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=config.lr))
volume_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=config.lr))

for e in range(10000):
    # h = model.fit_generator(
    #     gen,
    #     config.samples_per_epoch,
    #     nb_epoch=1,
    #     verbose=1)

    X, y = next(gen)
    (loss, acc) = model.train_on_batch(X, y)
    print("model", loss, acc)

    X, y = next(volume_gen)
    (loss, acc) = volume_model.train_on_batch(X, y)
    print("volume_model", loss, acc)

    # fpr, p_threshold, fpr_list, p_list = eval_model(model, volume_model)
    # print("fpr", fpr, "std", np.std(fpr_list), "p_threshold", p_threshold)
    # history['loss'].append(h.history['loss'][0])
    # history['acc'].append(h.history['acc'][0])
    # history['fpr'].append(fpr)
    # history['p_threshold'].append(float(p_threshold))
    # history['p_list'].append([ float(x) for x in p_list])

    # model.save_weights(SNAP_PATH + run_id + '.{:04d}'.format(e) + '.h5')

    # with open(SNAP_PATH + run_id + '.log.json', 'w') as fh:
    #     json.dump(history, fh)
