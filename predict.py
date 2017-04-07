import data
import datagen
import net
import scipy.ndimage.interpolation
import numpy as np
import pandas as pd
import pickle
import sys


SNAP_PATH = '/mnt/data/snap/'

config_name = sys.argv[1]
config = importlib.import_module(config_name)

fold = int(sys.argv[2])

classifier_weights_file = sys.argv[3]

localizer_output_dir = sys.argv[4]

output_file = sys.argv[5]


def ndsb17_get_predicted_nodules_v2(vsize, patient_ids, min_activity=30):
    X_nodules = []
    for pid in patient_ids:
        try:
            with open(SNAP_PATH + localizer_output_dir + 'boxes/' + pid + '.pkl', 'rb') as fh:
                label_boxes, label_sizes, label_activities_sum, label_activities_max = pickle.load( fh )
        except FileNotFoundError as e:
            print(pid, str(e))
            X_nodules.append(None)
            continue

        idx = np.argsort(label_activities_sum)[::-1][:1]
        box = label_boxes[idx]
        if box is None:
            print(pid, 'no areas')
            X_nodules.append(None)
            continue
        if label_activities_sum[idx] < min_activity: # TODO soft threshold
            print(pid, 'no areas with high enough activity', label_activities_sum[idx])
            X_nodules.append(None)
            continue
        
        center = 2*np.asarray([(box[0].start+box[0].stop)//2, (box[1].start+box[1].stop)//2, (box[2].start+box[2].stop)//2 ])
        diam = 2*np.mean([(box[0].start-box[0].stop), (box[1].start-box[1].stop), (box[2].start-box[2].stop) ])

        image = data.ndsb17_get_image(pid)
        #segmented_image = ndsb17_get_segmented_image(pid)

        pos = center - vsize//2
        volume = image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2] ]
        #segmented_volume = segmented_image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2] ]
        if volume.shape != (64,64,64):
            print(pid, 'shape mismatch')
            X_nodules.append(None)
            continue # TODO report something
        #volume = (volume + 1000)*segmented_volume - 1000
        X_nodules.append(volume)

    for n in range(len(X_nodules)):
        if X_nodules[n] is None:
            X_nodules[n] = np.zeros((64,64,64)) - 1000

    X_nodules = np.stack(X_nodules)[:,16:16+32,16:16+32,16:16+32,None]
    X_nodules = datagen.preprocess(X_nodules)
    X_nodules = scipy.ndimage.interpolation.zoom(X_nodules, (1, 0.5, 0.5, 0.5, 1), order=1)

    return X_nodules



df = data.ndsb17_get_df_test_labels()

vsize64 = np.asarray((64,64,64))

patient_ids = df["id"].tolist()

X_nodules = ndsb17_get_predicted_nodules_v2(vsize64, patient_ids, min_activity=10)

# p_base = len(df[df["cancer"]==1]) / len(df)

# y_true, y_pred = [], []
# for n in range(len(df)):
#     pid = df["id"][n]
#     #print(pid, df["cancer"][n])
#     y_true.append(df["cancer"][n])

y_true = df["cancer"].tolist()
y_true = np.asarray(y_true)


import importlib
import net

config_name = 'config_baseline2'
config = importlib.import_module(config_name)
model = net.model3d((16, 16, 16), sz=config.feature_sz, alpha=config.feature_alpha)
model.load_weights('/mnt/data/snap/config_baseline2__20170401054549.0023.h5')

y_test = model.predict(X_nodules, batch_size=64)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1., solver='lbfgs')

y_true = np.asarray(y_true)
clf.fit(y_test[:,None], y_true)

patient_ids_predict = patient_ids # TODO read from separate file

X_nodules_predict = ndsb17_get_predicted_nodules_v2(vsize64, patient_ids_predict, min_activity=10)

y_test = model.predict(X_nodules_predict, batch_size=64)

y_cal = clf.predict_proba(y_test[:,None])

for n in range(len(patient_ids_test)):
    print(patient_ids_test[n], y_cal[n])

