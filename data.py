import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from multiprocessing import Pool

from skimage import measure, morphology

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    binary_image = np.array(image > -320, dtype=np.int8)
    binary_image = scipy.ndimage.morphology.binary_dilation(binary_image, border_value=0, iterations=2).astype(np.int8)
    # HACK deal with trays
    binary_image[-1,:,0] = 0
    binary_image[-1,:,-1] = 0
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image += 1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image


# Load the scans in given folder path
def load_scans(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    # slices.sort(key = lambda x: int(x.InstanceNumber))
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array.astype(np.float32) * s.RescaleSlope + s.RescaleIntercept for s in scans])
    image = image.astype(np.int16)
   
    return image


import json


INPUT_FOLDER = '/mnt/data/ndsb17/unpacked/stage1/'
OUTPUT_FOLDER = '/mnt/data/ndsb17/unpacked/processed/'


def process_series(pid):
    print(pid)
    scans = load_scans(INPUT_FOLDER + pid)
    image = get_pixels_hu(scans)
    np.save(OUTPUT_FOLDER + 'original_resolution/' + pid + '.npy', image)

    spacing = [ float(x) for x in ([scans[0].SliceThickness] + scans[0].PixelSpacing) ]
    #print(spacing)

    with open(OUTPUT_FOLDER + 'original_resolution/' + pid + '.info.json', 'w') as f:
        json.dump({'spacing': spacing}, f)

    image_1mm, _ = resample(image, scans, [1,1,1])
    np.save(OUTPUT_FOLDER + '1mm/' + pid + '.npy', image_1mm)

    #     image_iso, _ = resample(image, scans, [spacing[1], spacing[1], spacing[2]])
    #     np.save(OUTPUT_FOLDER + 'iso/' + pid + '.npy', image_iso)

    segmented_lungs_fill = segment_lung_mask(image, True)
    np.save(OUTPUT_FOLDER + 'segmented_lungs/' + pid + '.npy', segmented_lungs_fill)


def is_segmentation_ok(pid):
    image = np.load(OUTPUT_FOLDER + 'segmented_lungs/' + pid + '.npy')
    if np.amax(image[:,256,:]) == 0:
        return False
    else:
        return True


# patient_ids = os.listdir(INPUT_FOLDER)
# patient_ids.sort()

# pool = Pool(processes=32)
# pool.imap_unordered(process_series, patient_ids)

LUNA16_PATH = '/mnt/data2/luna16/'

import pandas as pd
import glob


def luna16_get_df_nodes():
    df_nodes = pd.read_csv(LUNA16_PATH+"annotations.csv")
    df_nodes["pid"] = df_nodes["seriesuid"]
    return df_nodes


def luna16_get_patient_ids():
    patient_ids = glob.glob(LUNA16_PATH+"processed/images_1mm/*.npy")
    patient_ids = [ x.replace(LUNA16_PATH+"processed/images_1mm/", "").replace(".npy", "") for x in patient_ids ]
    return patient_ids


def luna16_get_image(pid):
    image = np.load(LUNA16_PATH + "processed/images_1mm/" + pid + ".npy", mmap_mode='r')
    return image


def luna16_get_segmented_image(pid):
    segmented_image = np.load(LUNA16_PATH + "processed/segmented_1mm/" + pid + ".npy", mmap_mode='r')
    return segmented_image


def luna16_get_info(pid):
    with open(LUNA16_PATH + 'processed/infos/' + pid + '.info.json') as f:
        info = json.load(f)
    return info


def luna16_get_volume(image, segmented_image, vsize, min_overlap=0.2):
    for n in range(100):
        pos = np.asarray([ np.random.randint(k, image.shape[k] - vsize[k]) for k in range(3) ])
        volume = image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
        segmented_volume = segmented_image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
        overlap = np.mean(segmented_volume)
        if overlap >= min_overlap:
            return volume, segmented_volume, overlap
    return None, None, None


def luna16_get_node_volume(image, vsize, info, df, idx, do_segmented_volume=False):
    node_x = df["coordX"].values[idx]
    node_y = df["coordY"].values[idx]
    node_z = df["coordZ"].values[idx]
    diam = df["diameter_mm"].values[idx]

    center = np.array([node_z,node_y,node_x])  #nodule center
    origin = np.array(info["origin"]) #x,y,z  Origin in world coordinates (mm)
    spacing = np.array(info["spacing_1mm"])# spacing of voxels in world coor. (mm)
    # c =np.rint((center-origin)/spacing)  # nodule center in voxel space
    # c = c.astype(np.int)
    pos = ((center-origin)/spacing - vsize/2)
    pos = np.rint(pos).astype(np.int)

    volume = image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
    return volume


def luna16_get_all_nodules(vsize, df_nodes):
    X = []
    diams = []
    for idx in range(len(df_nodes)):
        #print(idx)
        pid = df_nodes.iloc[idx]["pid"]
        image = luna16_get_image(pid)
        segmented_image = luna16_get_segmented_image(pid)
        info = luna16_get_info(pid)
        volume = luna16_get_node_volume(image, vsize, info, df_nodes, idx)
        X.append(volume.copy())
        diams.append(df_nodes.iloc[idx]["diameter_mm"])
    return X, diams


NDSB17_PATH = '/mnt/data2/ndsb17/'


# def ndsb17_get_df_nodes():
#     df = pd.read_csv(NDSB17_PATH + 'processed/annotations/' + 'stage1_annotations_v1.csv')
#     # TODO replace pids
#     return df_nodes


def ndsb17_get_df_labels():
    df = pd.read_csv(NDSB17_PATH + 'original/' + 'stage1_labels.csv')
    return df


def ndsb17_get_df_test_labels():
    df = pd.read_csv(NDSB17_PATH + 'original/' + 'stage1_test_labels.csv')
    return df


def ndsb17_get_patient_ids():
    patient_ids = glob.glob(NDSB17_PATH+"processed/images_1mm/*.npy")
    patient_ids = [ x.replace(NDSB17_PATH+"processed/images_1mm/", "").replace(".npy", "") for x in patient_ids ]
    return patient_ids


def ndsb17_get_patient_ids_noncancer():
    df_labels = ndsb17_get_df_labels()
    all_noncancer_pids = list(df_labels[df_labels["cancer"] == 0]["id"])
    return all_noncancer_pids


def ndsb17_get_df_nodes():
    df = pd.read_csv(NDSB17_PATH + 'processed/annotations/' + 'stage1_annotations_v2.csv')

    patient_ids = ndsb17_get_patient_ids()
    short_pid_to_pid = { x[:6]: x for x in patient_ids }
    
    for n in range(len(df)):
        tmp = df.ix[n, "pid"]
        if len(str(tmp)) == 6:
            short_pid = tmp
        if short_pid in short_pid_to_pid:
            df.ix[n, "pid"] = short_pid_to_pid[ short_pid ]
    for n in range(len(df)):
        df.ix[n, "diameter_mm"] = 10*df.ix[n, "Size, cm"]

    # filter by cancer only
    # TODO make this a parameter
    df_labels = ndsb17_get_df_labels()
    all_cancer_pids = list(df_labels[df_labels["cancer"] == 1]["id"])
    df = df[ df["pid"].isin(all_cancer_pids) ]

    return df


def ndsb17_get_image(pid):
    image = np.load(NDSB17_PATH + "processed/images_1mm/" + pid + ".npy", mmap_mode='r')
    return image


def ndsb17_get_segmented_image(pid):
    segmented_image = np.load(NDSB17_PATH + "processed/segmented_1mm/" + pid + ".npy", mmap_mode='r')
    return segmented_image


def ndsb17_get_info(pid):
    with open(NDSB17_PATH + 'processed/infos/' + pid + '.info.json') as f:
        info = json.load(f)
    return info


# def ndsb17_get_volume(image, segmented_image, vsize, min_overlap=0.2):
#     for n in range(100):
#         pos = np.asarray([ np.random.randint(k, image.shape[k] - vsize[k]) for k in range(3) ])
#         volume = image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
#         segmented_volume = segmented_image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
#         overlap = np.mean(segmented_volume)
#         if overlap >= min_overlap:
#             return volume, segmented_volume, overlap
#     return None, None, None


def ndsb17_get_node_volume(image, vsize, info, df, idx):
    # These positions are recoded as pixels in the native image

    x, y, z = df.iloc[idx]["X"], df.iloc[idx]["Y"], df.iloc[idx]["Im"]
    diam = df.iloc[idx]["Size, cm"] * 10

    # z seems to be flipped
    if info["do_flip_z"]:
        z = -z

    # x1 = x * info["spacing"][2] / info["spacing_1mm"][2]
    # y1 = y * info["spacing"][1] / info["spacing_1mm"][1]
    # z1 = z * info["spacing"][0] / info["spacing_1mm"][0]

    center = np.array([z, y, x])
    spacing = np.array(info["spacing"])
    spacing_1mm = np.array(info["spacing_1mm"])

    pos = (center * spacing/spacing_1mm) - (vsize/2)
    pos = np.rint(pos).astype(np.int)

    volume = image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
    return volume, diam


def ndsb17_get_all_nodules(vsize, df_nodes):
    X = []
    diams = []
    for idx in range(len(df_nodes)):
        try:
            #print(idx)
            pid = df_nodes.iloc[idx]["pid"]
            if pid == "b8bb02d229361a623a4dc57aa0e5c485":
                continue
            image = ndsb17_get_image(pid)
            #segmented_image = ndsb17_get_segmented_image(pid)
            info = ndsb17_get_info(pid)
            volume, diam = ndsb17_get_node_volume(image, vsize, info, df_nodes, idx)
            X.append(volume.copy())
            diams.append(diam)
        except FileNotFoundError as e:
            print(pid, repr(e))
    return X, diams



from scipy import signal


def compose_make_mask_gaussian(vsize, sigma=10):
    mask = signal.gaussian(vsize[0], std=sigma)[:,None,None] * \
        signal.gaussian(vsize[1], std=sigma)[None,:,None] * \
        signal.gaussian(vsize[2], std=sigma)[None,None,:]
    return mask


def compose_make_mask(vsize, diam, sigma):
    # mask = np.zeros(vsize, dtype=np.float32)
    # for i in range(vsize[0]):
    #     for j in range(vsize[1]):
    #         for k in range(vsize[2]):
    #             dist = np.sqrt(np.sum(np.square(np.asarray([i, j, k]) - vsize / 2.0)))
    #             if dist < diam/2.0:
    #                 mask[i,j,k] = 1

    grid = np.indices(vsize).astype(np.float32)
    grid = grid - vsize[:,None,None,None]/2.0
    mask = np.sqrt(np.sum(np.square(grid), axis=0)) < diam/2
                    
    mask = scipy.ndimage.filters.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return mask

def compose_max(volume, nodule, mask):
    x = np.amax( np.stack(((volume+1000) * (1-mask), (nodule+1000) * mask)), axis=0 ) - 1000
    return x


def compose_max2(volume, nodule, mask):
    x = np.amax( np.stack(((volume+1000), (nodule+1000) * mask)), axis=0 ) - 1000
    return x


def compose_mean(volume, nodule, mask):
    x = volume * (1-mask) + nodule * mask
    return x
