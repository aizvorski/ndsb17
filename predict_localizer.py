import numpy as np
import pandas as pd
import scipy
import skimage.transform
import pickle
import os
import sys
import importlib
import multiprocessing

import data
import datagen

SNAP_PATH = '/mnt/data/snap/'

config_label_threshold = 2

def predict_localizer(pid):
    global volume_model
    import net

    # global gpu_id
    print(pid)

    image = data.ndsb17_get_image(pid)
    segmented_image = data.ndsb17_get_segmented_image(pid)
    image = datagen.preprocess(image)

    image_2mm = skimage.transform.downscale_local_mean(image, (2,2,2), clip=False)
    segmented_image_2mm = skimage.transform.downscale_local_mean(segmented_image.astype(np.float32), (2,2,2), clip=False)
    segmented_image_2mm = (segmented_image_2mm > 0)

    predicted_image = net.tiled_predict(volume_model, image_2mm)
    np.save(SNAP_PATH + localizer_output_dir + 'predicted_image/' + pid + '.npy', predicted_image)

    predicted_masked = predicted_image * segmented_image_2mm

    selem = np.ones((3,3,3), dtype=int)
    labeled_array, num_features = scipy.ndimage.measurements.label( predicted_masked > config_label_threshold, structure=selem )
    label_boxes = scipy.ndimage.measurements.find_objects(labeled_array)
    label_sizes = scipy.ndimage.measurements.sum(np.ones(predicted_masked.shape, dtype=int), labeled_array, index=range(num_features+1))
    label_activities_sum = scipy.ndimage.measurements.sum(predicted_masked, labeled_array, index=range(num_features+1))
    label_activities_max = scipy.ndimage.measurements.maximum(predicted_masked, labeled_array, index=range(num_features+1))
    label_boxes = [None] + label_boxes

    for idx in np.argsort(label_activities_sum)[::-1][:5]:
        print(idx, label_boxes[idx], label_sizes[idx], label_activities_sum[idx])

    with open(SNAP_PATH + localizer_output_dir + 'boxes/' + pid + '.pkl', 'wb') as fh:
        pickle.dump( (label_boxes, label_sizes, label_activities_sum, label_activities_max), fh )

    return predicted_image, (label_boxes, label_sizes, label_activities_sum, label_activities_max)


def predict_localizer_for_map(pid):
    # if os.path.exists('/mnt/data/ndsb17/predict/boxes/' + pid + '.pkl'):
    #     return

    predict_localizer(pid)
    # no return value


def gpu_init(queue):
    global gpu_id
    gpu_id = queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import tensorflow as tf

    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    process = multiprocessing.current_process()
    print(gpu_id, process.pid, gpu_list)

    global volume_model
    import net
    volume_model = net.model3d((64, 64, 64), sz=config.feature_sz, alpha=config.feature_alpha, do_features=True)
    volume_model.load_weights(weights_file)


# def f(x):
#     global gpu_id

#     process = multiprocessing.current_process()
#     return (gpu_id, process.pid, x * x)


if __name__ == "__main__":
    config_name = sys.argv[1]
    config = importlib.import_module(config_name)

    weights_file = sys.argv[2]

    localizer_output_dir = sys.argv[3]

    manager = multiprocessing.Manager()
    gpu_init_queue = manager.Queue()

    num_gpu = 2
    for i in range(num_gpu):
        gpu_init_queue.put(i)

    p = multiprocessing.Pool(num_gpu, gpu_init, (gpu_init_queue,))


    patient_ids = data.ndsb17_get_patient_ids()

    p.map(predict_localizer_for_map, patient_ids)
