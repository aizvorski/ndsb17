import pyflann
import numpy as np
import data
import skimage.transform
import flann

import data
import datagen

def featurize(v, b=1):
    sides = []
    sides += [ v[:b,:,:] ]
    sides += [ v[-b:,:,:] ]
    sides += [ v[b:-b,:b,:] ]
    sides += [ v[b:-b,-b:,:] ]
    sides += [ v[b:-b,b:-b,:b] ]
    sides += [ v[b:-b,b:-b,-b:] ]
    return np.concatenate([ x.ravel() for x in sides ]).astype(np.float32)


def index_image(image, sc=4, sz=8):
    image_lowres = skimage.transform.downscale_local_mean(image, (sc,sc,sc), clip=False)
    patch_coords = []
    features = []
    for i in range(image_lowres.shape[0]-sz):
        for j in range(image_lowres.shape[1]-sz):
            for k in range(image_lowres.shape[2]-sz):
                patch_coords.append((i,j,k))
                f = featurize(image_lowres[i:i+sz,j:j+sz,k:k+sz])
                features.append(f)
    features = np.asarray(features, dtype=np.float32)
    fidx = pyflann.FLANN()
    print(features.shape)
    fidx.build_index(features)
    return fidx, patch_coords


def lookup_image(fidx, volume, sc=4, sz=8, n=5):
    volume_lowres = skimage.transform.downscale_local_mean(volume, (sc,sc,sc), clip=False)
    f = featurize(volume_lowres)
    indices, distances = fidx.nn_index(f[None,:], n)
    return indices[0]


def get_match(image, patch_coords, idx, sc=4, sz=8):
    i,j,k = patch_coords[idx]
    return image[i*sc:(i+sz)*sc, j*sc:(j+sz)*sc, k*sc:(k+sz)*sc]


def sample_generator_ab(vsize, patient_ids, X_nodules_a, diams_a, X_nodules_b, diams_b):
    while True:
        patient_ids = np.random.permutation(patient_ids)
        for pid in patient_ids:
            fidx, patch_coords = index_image(image)

            for k in range(100):
                if np.random.choice([True,False]):
                    n = np.random.choice(len(X_nodules_a))
                    nodule = X_nodules_a[n]
                    diam = diams_a[n]
                else:
                    n = np.random.choice(len(X_nodules_b))
                    nodule = X_nodules_b[n]
                    diam = diams_b[n]

                nodule = datagen.volume_flip(nodule)
                nodule = datagen.volume_rotation(nodule, np.random.randint(0,360))
                nodule = datagen.volume_crop(nodule, vsize)

                indices = lookup_image(fidx, nodule, n=10)
                idx = np.random.choice(indices)

                volume = get_match(image, patch_coords, idx)
                mask = data.compose_make_mask(vsize, diam=diam+6, sigma=(diam+6)/8)
                volume_aug = data.compose_max2(volume, nodule, mask)

                yield(volume_aug)


def batch_generator_ab(vsize, patient_ids, X_nodules_a, diams_a, X_nodules_b, diams_b):
