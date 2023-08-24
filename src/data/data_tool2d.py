import shutil
import sys

from tqdm import tqdm

from src.utils.util_contour import get_slices_and_mask

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import yaml
import numpy as np
import pandas as pd
import glob
import pydicom
from src.utils import util_data, util_dicom, util_path
import dicom2nifti
import nibabel as nib



import argparse
argparser = argparse.ArgumentParser(description='Prepare data for training')
argparser.add_argument('-c', '--config',
                       help='configuration file path', default='./configs/prepare_data2d_RC.yaml')
args = argparser.parse_args()






if __name__ == '__main__':

    # Config file
    print("Upload configuration file")
    config_file = args.config



    with open(config_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Parameters
    dataset_name = cfg['data']['dataset_name']
    img_dir = cfg['data']['img_dir']
    label_file = cfg['data']['label_file']

    interim_dir =os.path.join(cfg['data']['interim_dir'], dataset_name)
    processed_dir = os.path.join(cfg['data']['processed_dir'], dataset_name)
    reports_dir = os.path.join(cfg['reports']['reports_dir'], dataset_name)
    img_dir = os.path.join(img_dir, dataset_name)

    # Patient Information Dicom
    patients_info_file = os.path.join(interim_dir, 'patients_info.xlsx')
    patients_info_df = pd.read_excel(patients_info_file).set_index('PatientID')

    # RTstruct file

    rtstruct_file = os.path.join(interim_dir, 'structures.xlsx')
    rtstruct_df = pd.read_excel(rtstruct_file).set_index('patient_dir')


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

            # Create patient folders

            patient_folders = util_path.make_patient_folders(patient_fname, processed_dir, dataset_name, extension='tiff', force=False, create=True)

            # Get ROIs from segmentation file
            struct_info = eval(rtstruct_df.loc[patient_dir]['structures'])

            # filter ROIS:
            #struct_info = filter_rois(struct_info, cfg['data']['rois_to_filter'])

            img_voxel, metadatas, rois_dict = get_slices_and_mask(ds_seg, roi_names=list(struct_info.values()),
                                                                  patient_dir=patient_dir)




            # Iterate all the patient slices
            for dicom_file_path in dicom_files:


                # Read dicom file
                slice_dicom = pydicom.dcmread(dicom_file_path)


                # Get slice number
                slice_number = slice_dicom.InstanceNumber

                # Get Patient Information:
                patient_info = patients_info_df.loc[patient_fname]

                # Get slice intercept
                slice_intercept, slice_slope = patient_info['RescaleIntercept'], patient_info['RescaleSlope']

                # Pixel array
                slice_pixel_array = slice_dicom.pixel_array

                # Rescale to HU and padding air
                slice_HU = transform_to_HU(slice=slice_pixel_array,
                                           intercept=slice_intercept,
                                           slope=slice_slope,
                                           padding_value=cfg['data']['preprocessing']['range']['min'],
                                           change_value="lower",
                                           new_value=cfg['data']['preprocessing']['range']['min'])

                min_slice_pixel, max_slice_pixel = np.min(slice_HU), np.max(slice_HU)
                # CLIPPING: Clip the values to the range [min, max]




                # Get slice thickness
                slice_thickness = slice_dicom.SliceThickness

                # Get slice location
                slice_location = slice_dicom.SliceLocation

                # Get slice position
                slice_position = slice_dicom.ImagePositionPatient

                # Get slice orientation
                slice_orientation = slice_dicom.ImageOrientationPatient

                # Get slice pixel spacing
                slice_pixel_spacing = slice_dicom.PixelSpacing






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
