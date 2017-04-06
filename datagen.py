import data
import numpy as np
import random
import scipy.ndimage.interpolation
import skimage.transform


# X_mean, X_std = -378.9, 475.913 # for LUNA16
X_mean, X_std = -298.099, 436.168


def preprocess(image):
    return (image - X_mean) / X_std


def volume_rotation(volume, angle):
    result = np.zeros(volume.shape, dtype=np.float32)
    for i in range(volume.shape[0]):
        result[i] = skimage.transform.rotate(volume[i], angle, resize=False, preserve_range=True, order=1)
    return result


def volume_crop(volume, vsize):
    p0 = (volume.shape[0] - vsize[0])//2
    p1 = (volume.shape[1] - vsize[1])//2
    p2 = (volume.shape[2] - vsize[2])//2
    return volume[p0:p0+vsize[0], p1:p1+vsize[1], p2:p2+vsize[2] ]


def volume_flip(volume):
    if random.choice([True, False]):
        volume = volume[::-1,:,:]
    if random.choice([True, False]):
        volume = volume[:,::-1,:]
    if random.choice([True, False]):
        volume = volume[:,:,::-1]
    return volume


def make_augmented(vsize, volume, X_nodules, diams, do_flip=True, do_rotate=True):
    idx = random.choice(range(len(X_nodules)))
    nodule = X_nodules[idx]
    # randomly flip or not flip each axis
    if do_flip:
        nodule = volume_flip(nodule)
    if do_rotate:
        nodule = volume_rotation(nodule, np.random.randint(0,360))
    nodule = volume_crop(nodule, vsize)
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
            except FileNotFoundError as e:
                #print(pid, repr(e))
                continue
            
        pos = np.asarray([ np.random.randint(k, image.shape[k] - vsize[k]) for k in range(3) ])
        segmented_volume = segmented_image[pos[0]:pos[0]+vsize[0], pos[1]:pos[1]+vsize[1], pos[2]:pos[2]+vsize[2]]
        is_lung = True
        if np.count_nonzero(segmented_volume) == 0:
            is_lung = False
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
        if central_density < -500 and is_lung and np.random.choice([True, False]):
            volume = make_augmented(vsize, volume, X_nodules, diams)
            is_augmented = True
            n_aug += 1
            
        n+=1

        yield volume, is_augmented


def batch_generator(vsize, patient_ids, X_nodules, diams, batch_size=64, do_downscale=True):
    gen = sample_generator(vsize, patient_ids, X_nodules, diams)
    
    while True:
        X = np.zeros((batch_size, 32,32,32,1), dtype=np.float32)
        #y = np.zeros((batch_size, 2), dtype=np.int)
        y = np.zeros((batch_size), dtype=np.int)
        for n in range(batch_size//2):
            volume, is_augmented = next(gen)
            if not is_augmented:
                continue
            X[n,:,:,:,0] = volume
            y[n] = 1

        for n in range(batch_size//2, batch_size):
            volume, is_augmented = next(gen)
            if is_augmented:
                continue
            X[n,:,:,:,0] = volume
            y[n] = 0

        X = (X - X_mean)/X_std
        if do_downscale:
            X = skimage.transform.downscale_local_mean(X, (1,2,2,2,1), clip=False)
        yield X, y


def batch_generator_ab(vsize, patient_ids, X_nodules_a, diams_a, X_nodules_b, diams_b, batch_size=64, do_downscale=True):
    gen_a = sample_generator(vsize, patient_ids, X_nodules_a, diams_a)
    gen_b = sample_generator(vsize, patient_ids, X_nodules_b, diams_b)
    
    while True:
        X = np.zeros((batch_size, 32,32,32,1), dtype=np.float32)
        #y = np.zeros((batch_size, 2), dtype=np.int)
        y = np.zeros((batch_size), dtype=np.int)
        n = 0
        while n < batch_size:
            if np.random.random() < 0.5:
                volume, is_augmented = next(gen_a)
                if not is_augmented: 
                    continue
                #y[n,0] = 1
            else:
                volume, is_augmented = next(gen_b)
                if not is_augmented: 
                    continue
                #y[n,1] = 1
                y[n] = 1
            X[n,:,:,:,0] = volume
            n += 1
        X = (X - X_mean)/X_std
        if do_downscale:
            X = skimage.transform.downscale_local_mean(X, (1,2,2,2,1), clip=False)
        yield X, y
