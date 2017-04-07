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

model = net.model3d(config.net_input_vsize, sz=config.feature_sz, alpha=config.feature_alpha)
model.load_weights(SNAP_PATH + classifier_weights_file, by_name=True)

df = data.ndsb17_get_df_test_labels()
p_base = len(df[df["cancer"]==1]) / len(df)

vsize64 = np.asarray((64,64,64))

def predict_classifier(patient_ids):
    X_nodules, predicted_patient_ids = data.ndsb17_get_predicted_nodules(vsize64, patient_ids, SNAP_PATH + localizer_output_dir, min_activity=config.min_activity_predict)
    X_nodules = np.stack(X_nodules)[:,16:16+32,16:16+32,16:16+32,None]
    X_nodules = datagen.preprocess(X_nodules)
    if config.do_downscale:
        X_nodules = skimage.transform.downscale_local_mean(X_nodules, (1,2,2,2,1), clip=False)

    y_pred = model.predict(X_nodules, batch_size=64)[:,0]

    max_y_by_pid = {}
    for n in range(len(predicted_patient_ids)):
        pid = predicted_patient_ids[n]
        if not pid in max_y_by_pid:
            max_y_by_pid[ pid ] = y_pred[n]
        max_y_by_pid[ pid ] = max( y_pred[n], max_y_by_pid[ pid ] )

    y_pred = []
    for n in range(len(patient_ids)):
        if pid in max_y_by_pid:
            y_pred.append(max_y_by_pid[ pid ])
        else:
            y_pred.append(0)

    return y_pred

patient_ids_test = df["id"].tolist()

y_test = predict_classifier(patient_ids_test)
y_test = np.asarray(y_test)

y_true = df["cancer"].tolist()
y_true = np.asarray(y_true)

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

y_predict = predict_classifier(patient_ids)
y_predict_calibrated = clf.predict_proba(y_predict[:,None])

# for n in range(len(patient_ids_predict)):
#     print(patient_ids_predict[n], y_predict_calibrated[n,1])

df_output = pd.DataFrame({'id': patient_ids_predict, 'cancer': y_predict_calibrated[:,1]}, columns=['id', 'cancer'])
df_output.to_csv(output_file)
