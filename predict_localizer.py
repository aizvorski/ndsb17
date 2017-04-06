import data
import datagen
import net
import skimage.transform
import numpy as np
import pandas as pd
import pickle
import os
import sys
import importlib
import scipy

config_name = sys.argv[1]
config = importlib.import_module(config_name)

weights_file = sys.argv[2]

#volume_model = net.model3d((64, 64, 64), sz=config.feature_sz, alpha=config.feature_alpha, do_features=True)
volume_model = net.model3d((64, 64, 64), sz=32, alpha=1.5, do_features=True)
volume_model.load_weights(weights_file)

df = data.ndsb17_get_df_labels()

config_label_threshold = 2

def predict_regions(pid):
    # if os.path.exists('/mnt/data/ndsb17/predict/boxes/' + pid + '.pkl'):
    #     return

    image = data.ndsb17_get_image(pid)
    segmented_image = data.ndsb17_get_segmented_image(pid)
    image = datagen.preprocess(image)

    image_2mm = skimage.transform.downscale_local_mean(image, (2,2,2), clip=False)
    segmented_image_2mm = skimage.transform.downscale_local_mean(segmented_image.astype(np.float32), (2,2,2), clip=False)
    segmented_image_2mm = (segmented_image_2mm > 0)

    predicted_image = net.tiled_predict(volume_model, image_2mm)[:,:,:,0]
    np.save('/mnt/data/ndsb17/predict/' + pid + '.npy', predicted_image)

    predicted_image *= segmented_image_2mm

    selem = np.ones((3,3,3), dtype=int)
    labeled_array, num_features = scipy.ndimage.measurements.label( predicted_image > config_label_threshold, structure=selem )
    label_boxes = scipy.ndimage.measurements.find_objects(labeled_array)
    label_sizes = scipy.ndimage.measurements.sum(np.ones(predicted_image.shape, dtype=int), labeled_array, index=range(num_features+1))
    label_activities_sum = scipy.ndimage.measurements.sum(predicted_image, labeled_array, index=range(num_features+1))
    label_activities_max = scipy.ndimage.measurements.maximum(predicted_image, labeled_array, index=range(num_features+1))
    label_boxes = [None] + label_boxes

    for idx in np.argsort(label_activities_sum)[::-1][:5]:
        print(idx, label_boxes[idx], label_sizes[idx], label_activities_sum[idx])

    with open('/mnt/data/ndsb17/predict/boxes/' + pid + '.pkl', 'wb') as fh:
        pickle.dump( (label_boxes, label_sizes, label_activities_sum, label_activities_max), fh )

for n in range(len(df)):
    pid = df["id"][n]
    print(pid, df["cancer"][n])

    try:
        predict_regions(pid)
    except Exception as e:
        print(str(e))
