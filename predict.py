import data
import datagen
import net
import scipy.ndimage.interpolation
import skimage.transform
import numpy as np
import pandas as pd
import pickle
import sys
import importlib
import net


SNAP_PATH = '/mnt/data/snap/'

config_name = sys.argv[1]
config = importlib.import_module(config_name)

fold = int(sys.argv[2])

classifier_weights_file = sys.argv[3]

localizer_output_dir = sys.argv[4]

patient_ids_predict_file = sys.argv[5]

output_file = sys.argv[6]


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
    X_nodules = skimage.transform.downscale_local_mean(X_nodules, (1,2,2,2,1), clip=False)
    return X_nodules



df = data.ndsb17_get_df_test_labels()

vsize64 = np.asarray((64,64,64))

patient_ids = df["id"].tolist()

X_nodules = ndsb17_get_predicted_nodules_v2(vsize64, patient_ids, min_activity=config.min_activity_predict)

# p_base = len(df[df["cancer"]==1]) / len(df)

# y_true, y_pred = [], []
# for n in range(len(df)):
#     pid = df["id"][n]
#     #print(pid, df["cancer"][n])
#     y_true.append(df["cancer"][n])

y_true = df["cancer"].tolist()
y_true = np.asarray(y_true)


model = net.model3d((16, 16, 16), sz=config.feature_sz, alpha=config.feature_alpha)
model.load_weights(SNAP_PATH + classifier_weights_file)

y_test = model.predict(X_nodules, batch_size=64)[:,0]
print("y_test", y_test.shape)
print("y_true", y_true.shape)


from sklearn.linear_model import LogisticRegression
import sklearn.metrics

clf = LogisticRegression(C=1., solver='lbfgs')

clf.fit(y_test[:,None], y_true)

y_test_calibrated = clf.predict_proba(y_test[:,None])

print("log loss", sklearn.metrics.log_loss(y_true, y_test_calibrated))

df_patient_ids_predict = pd.read_csv(patient_ids_predict_file)
patient_ids_predict = df_patient_ids_predict["id"].tolist()

X_nodules_predict = ndsb17_get_predicted_nodules_v2(vsize64, patient_ids_predict, min_activity=config.min_activity_predict)

y_predict = model.predict(X_nodules_predict, batch_size=64)[:,0]

y_predict_calibrated = clf.predict_proba(y_predict[:,None])

# for n in range(len(patient_ids_predict)):
#     print(patient_ids_predict[n], y_predict_calibrated[n,1])

df_output = pd.DataFrame({'id': patient_ids_predict, 'cancer': y_predict_calibrated[:,1]}, columns=['id', 'cancer'])
df_output.to_csv(output_file)
