import pandas as pd
import numpy as np
import SimpleITK as sitk
import scipy.ndimage.interpolation
import skimage.transform

luna_path = "/mnt/data/luna16/unpacked/"

# read metadata

from glob import glob

def get_df_node():
    global file_list
    
    df_node = pd.read_csv(luna_path+"annotations.csv")

    file_list=glob(luna_path+"*/*.mhd")

    def get_filename(case):
        global file_list
        for f in file_list:
            if case in f:
                return(f)

    df_node["file"] = df_node["seriesuid"].apply(get_filename)
    return df_node

def read_nodes(img_file, df_node, sz=32):
    # img_file = "luna16_unpack/subset2/1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405.mhd"
    itk_img = sitk.ReadImage(img_file)
    img_array = sitk.GetArrayFromImage(itk_img)

    origin = np.array(itk_img.GetOrigin()) #x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())# spacing of voxels in world coor. (mm)

    z_scale = spacing[2]/spacing[1]
    img_iso = scipy.ndimage.interpolation.zoom(img_array, zoom=(z_scale, 1,1), order=1)

    img_nodes = []
    diams = []

    mini_df = df_node[df_node["file"]==img_file]
    for idx in range(len(mini_df)):
        node_x = mini_df["coordX"].values[idx]
        node_y = mini_df["coordY"].values[idx]
        node_z = mini_df["coordZ"].values[idx]
        diam = mini_df["diameter_mm"].values[idx]

        center = np.array([node_x,node_y,node_z])  #nodule center
        v_center =np.rint((center-origin)/spacing)  # nodule center in voxel space
        c = v_center.astype(np.int)

        img_node = img_iso[int(z_scale*c[2]-sz):int(z_scale*c[2]+sz), c[1]-sz:c[1]+sz, c[0]-sz:c[0]+sz]

        img_nodes.append(img_node)
        diams.append(diam / spacing[0])
        # TODO return spacing here as well

    return img_nodes, diams

def get_node_sections(img_node, diam):
    return


def load_scans_luna16(img_file):
    itk_img = sitk.ReadImage(img_file)
    img_array = sitk.GetArrayFromImage(itk_img)

    origin = np.array(itk_img.GetOrigin()) #x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())# spacing of voxels in world coor. (mm)

    return itk_img, spacing, origin

def resample_luna16(image, spacing, new_spacing):
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=1)
    
    return image, new_spacing

import json

INPUT_FOLDER = '/mnt/data/luna16/unpacked/'
OUTPUT_FOLDER = '/mnt/data/luna16/processed/'

def process_series(pid):
    print(pid)
    # input_file = glob(INPUT_FOLDER+"*/" +pid+  ".mhd")[0]
    image, spacing, origin = load_scans_luna16(INPUT_FOLDER+"/" +pid+  ".mhd")
    np.save(OUTPUT_FOLDER + 'original_resolution/' + pid + '.npy', image)

    with open(OUTPUT_FOLDER + 'original_resolution/' + pid + '.info.json', 'w') as f:
        json.dump({'spacing': list(spacing), 'origin':list(origin)}, f)

    image_1mm, _ = resample(image, scans, [1,1,1])
    np.save(OUTPUT_FOLDER + '1mm/' + pid + '.npy', image_1mm)

    #     image_iso, _ = resample(image, scans, [spacing[1], spacing[1], spacing[2]])
    #     np.save(OUTPUT_FOLDER + 'iso/' + pid + '.npy', image_iso)

    segmented_lungs_fill = segment_lung_mask(image, True)
    np.save(OUTPUT_FOLDER + 'segmented_lungs/' + pid + '.npy', segmented_lungs_fill)



df_node = get_df_node()
files = list(set(df_node["file"]))

# all_diams = {}

# n = 0
# for img_file in files:
#     print img_file
#     img_nodes, diams = read_nodes(img_file, df_node)
#     for i in range(len(img_nodes)):
#         print n
#         np.save("luna16_cube64/nodules/" + str(n) + ".npy", img_nodes[i])
#         all_diams[n] = diams[i]
#         n += 1

# all_diams_array = np.zeros((n,), dtype=int)
# for i in range(n):
#     all_diams_array[i] = all_diams[i]

# np.save("luna16_cube64/nodules/all_diams.npy", all_diams_array)

for pid in df_node["seriesuid"]:
    process_series(pid)
