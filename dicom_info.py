import dicom
dcm = 'sample_images/0a38e7597ca26f9374f8ea2770ba870d/4ec5ef19b52ec06a819181e404d37038.dcm'
print('Filename: {}'.format(dcm))
dcm = dicom.read_file(dcm)
print(str(dcm))
