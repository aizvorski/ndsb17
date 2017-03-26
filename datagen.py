import data
import numpy as np
import random
import scipy.ndimage.interpolation

# X_mean, X_std = -378.9, 475.913 # for LUNA16
X_mean, X_std = -298.099, 436.168


def make_augmented(vsize, volume, X_nodules, diams):
    idx = random.choice(range(len(X_nodules)))
    nodule = X_nodules[idx]
    # randomly flip or not flip each axis
    if random.choice([True, False]):
        nodule = nodule[::-1,:,:]
    if random.choice([True, False]):
        nodule = nodule[:,::-1,:]
    if random.choice([True, False]):
        nodule = nodule[:,:,::-1]
    mask = data.compose_make_mask(vsize, diam=diams[idx]+6, sigma=(diams[idx]+6)/8)
    volume_aug = data.compose_max2(volume, nodule, mask)
    return volume_aug


def sample_generator(vsize, patient_ids, X_nodules, diams):
    n = 0
    n_aug = 0

    central_mask = data.compose_make_mask(vsize, diam=6+6, sigma=(6+6)/8)
    
    while True:
        if n % 1000 == 0:
            try:
                pid = random.choice(patient_ids)
                image_ = data.ndsb17_get_image(pid)
                segmented_image_ = data.ndsb17_get_segmented_image(pid)

                image, segmented_image = image_, segmented_image_
                n+=1
                # segpack = np.packbits(segmented_image, axis=0)
                # info = data.luna16_get_info(pid)
            except Exception as e:
                #print(pid, repr(e))
                continue
            
        pos = np.asarray([ np.random.randint(k, image.shape[k] - vsize[k]) for k in range(3) ])
        segmented_volume = segmented_image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
        if np.count_nonzero(segmented_volume) == 0:
            if np.random.random() > 0.01:
                continue
#         segpack_volume = segpack[pos[0]//8:(pos[0]+vsize[0])//8, pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
#         if np.count_nonzero(segpack_volume) == 0:
#            continue
        volume = image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
#         overlap = np.mean(segmented_volume)
#         density = np.mean(volume)
        central_density = np.mean((volume+1000) * central_mask) / np.mean(central_mask) - 1000
    
        is_augmented = False
        if central_density < -500 and np.random.choice([True, False]):
            volume = make_augmented(vsize, volume, X_nodules, diams)
            is_augmented = True
            n_aug += 1
            
        n+=1

        yield volume, is_augmented


def batch_generator(vsize, patient_ids, X_nodules, diams):
    gen = sample_generator(vsize, patient_ids, X_nodules, diams)
    batch_size = 64
    while True:
        X = np.zeros((batch_size, 32,32,32,1), dtype=np.float32)
        y = np.zeros((batch_size, 2), dtype=np.int)
        for n in range(batch_size):
            volume, is_augmented = next(gen)
            X[n,:,:,:,0] = volume
            if is_augmented:
                y[n,1] = 1
            else:
                y[n,0] = 1
        X = (X - X_mean)/X_std
        X = scipy.ndimage.interpolation.zoom(X, (1, 0.5, 0.5, 0.5, 1), order=1)
        yield X, y
