import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam

from keras_tqdm import TQDMNotebookCallback

import data
import datagen
import net

#import importlib; importlib.reload(data)
import random
import scipy.ndimage.interpolation

vsize = np.asarray([32,32,32])

df_nodes = data.ndsb17_get_df_nodes() 
df_nodes = df_nodes[(df_nodes["diameter_mm"]>10)]

patient_ids = data.ndsb17_get_patient_ids()

X_nodules, diams = data.ndsb17_get_all_nodules(vsize, df_nodes)

gen = datagen.batch_generator(vsize, patient_ids, X_nodules[:-50], diams[:-50])


model = net.model3d((16, 16, 16))
print(model.summary())


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

batch_size=64

h = model.fit_generator(
    gen,
    10000,
    nb_epoch=100,
    verbose=1)

print(h.history)

model.save_weights('tmp2.h5')
