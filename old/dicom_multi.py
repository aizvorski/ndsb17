import dicom
import glob
import sys
import numpy as np

slice_locations = []
pixel_spacing = []
for f in glob.glob(sys.argv[1] + '/*.dcm'):
	dcm = dicom.read_file(f)
	slice_locations.append(dcm.SliceLocation)
	pixel_spacing.append(dcm.PixelSpacing[0])
slice_locations = sorted(slice_locations)
print np.median(np.diff(slice_locations))
print np.median(pixel_spacing)
