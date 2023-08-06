import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import yaml
import numpy as np
import pandas as pd
import glob
import pydicom
import dicom2nifti
import nibabel as nib

from src.utils import util_data
from src.utils import util_path


#### TODO: TEMPORARY FUNCTIONS ####
# TODO: move to utils

def set_padding_to_air(image, padding_value=-1000, change_value="lower", new_value=-1000):
    """
    This function sets the padding (contour of the dicom image) to the padding value. It trims all the values below the
    padding value and sets them to this value.

    Parameters
    ----------
    image: numpy.array
        Image to modify.
    padding_value: scalar number
        Value to use as threshold in the trim process and to be set in those points.
    change_value: string, says if the values to be changed are greater or lower than the padding value.
    new_value: scalar number, value to be set in the points where the change_value is True.

    Returns
    -------
    image: numpy.array
        Modified image.

    """

    trim_map = image < padding_value
    options = {"greater": ~trim_map, "lower": trim_map}
    image[options[change_value]] = new_value

    return image



def transform_to_HU(slice, intercept, slope, padding_value=-1000, change_value="lower", new_value=-1000):
    """
    This function transforms to Hounsfield units all the images passed.

    Parameters
    ----------
    slice: numpy.array
        List of metadatas of the slices where to gather the information needed in the transformation.
    intercept: scalar number
        Intercept of the slice.
    slope: scalar number
        Slope of the slice.
    padding_value: scalar number
        Value to use as threshold in the trim process and to be set in those points.
    change_value: string, says if the values to be changed are greater or lower than the padding value.
    new_value: scalar number, value to be set in the points where the change_value is True.

    Returns
    -------
    images: numpy array
        transformed slice in HU with the padding set to the padding value.
    """


    if slope != 1:

        slice = slope * slice.astype("float32")
        slice = slice.astype("float32")
        slice += np.float32(intercept)
        # TODO qui è necessario fare il riaggiustamento per i pazienti con l'intercept sballata @ltrochin
    # Padding to air all the values below
    slice_padded = set_padding_to_air(image=slice,
                                      padding_value=padding_value,
                                      change_value=change_value,
                                      new_value=new_value)
    return slice_padded




if __name__ == '__main__':

    # Config file
    print("Upload configuration file")
    with open('./configs/prepare_data2d.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Parameters
    dataset_name = cfg['data']['dataset_name']
    img_dir = cfg['data']['img_dir']
    label_file = cfg['data']['label_file']

    interim_dir =os.path.join(cfg['data']['interim_dir'], dataset_name)
    processed_dir = os.path.join(cfg['data']['processed_dir'], dataset_name)
    reports_dir = os.path.join(cfg['reports']['reports_dir'], dataset_name)

    preprocessing = cfg['data']['preprocessing']
    img_size = preprocessing['img_size']
    interpolation = preprocessing['interpolation']

    # List all patients in the source directory
    patients_list = glob.glob(os.path.join(img_dir, "*")) # patient folders

    for patient_dir in patients_list:

        if os.path.isdir(patient_dir):
            print("\n")
            print(f"Patient: {patient_dir}")

            # Load idpatient from dicom file
            dicom_files = glob.glob(os.path.join(patient_dir, 'CT*.dcm'))
            # Open files
            seg_files = glob.glob(os.path.join(patient_dir, 'RS*.dcm'))
            ds_seg = pydicom.dcmread(seg_files[0])

            # Select idpatient
            ds = pydicom.dcmread(dicom_files[0])
            patient_fname = getattr(ds, 'PatientID', None)
            assert patient_fname is not None, "Patient ID not found"


            # Iterate all the patient slices
            for dicom_file_path in dicom_files:
                # Read dicom file
                slice_dicom = pydicom.dcmread(dicom_file_path)








                pass





            # Adapt HU
            # todo @ltronchin


            # Adaptive pixel spacing interpolation2d

            # todo @fruffini




            # Resize

            # Normalization
            # todo clip hu
            # todo normalize in 0,1
            # todo @ltronchin

            # Save 2D/3D volume
            # todo @fruffini

    print("May the force be with you")
