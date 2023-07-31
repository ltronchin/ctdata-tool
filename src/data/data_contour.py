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
from src.utils import util_dicom

if __name__ == '__main__':

    # Config file
    print("Upload configuration file")
    with open('./configs/prepare_data3d.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Parameters
    dataset_name = cfg['data']['dataset_name']
    img_dir = cfg['data']['img_dir']
    label_file = cfg['data']['label_file']

    interim_dir =os.path.join(cfg['data']['interim_dir'], dataset_name)
    processed_dir = os.path.join(cfg['data']['processed_dir'], dataset_name)
    reports_dir = os.path.join(cfg['reports']['reports_dir'], dataset_name)
    reports_dir = os.path.join(reports_dir, 'contour')
    util_path.create_dir(reports_dir)

    preprocessing = cfg['data']['preprocessing']
    img_size = preprocessing['img_size']
    interpolation = preprocessing['interpolation']

    # List all patients in the source directory
    patients_list = glob.glob(os.path.join(img_dir, "*")) # patient folders

    # Structure dataframe

    data = []

    for patient_dir in patients_list:

        if os.path.isdir(patient_dir):
            print("\n")
            print(f"Patient: {patient_dir}")

            # Convert DICOM to NIFTI 3D volume
            interim_dir_nifti = os.path.join(interim_dir, 'nifti_volumes')
            util_path.create_dir(interim_dir_nifti)

            # Load idpatient from dicom file
            dicom_files = glob.glob(os.path.join(patient_dir, 'CT*.dcm'))
            ds = pydicom.dcmread(dicom_files[0])
            # Open files
            seg_files = glob.glob(os.path.join(patient_dir, 'RS*.dcm'))
            assert len(seg_files) <=1

            try:
                ds_seg = pydicom.dcmread(seg_files[0])
                idx = np.where(np.array(util_dicom.get_roi_names(ds_seg)) == "Polmone sx")[0][0]
                contour_datasets = util_dicom.get_roi_contour_ds(ds_seg, idx)
                mask_dict = util_dicom.get_mask_dict(contour_datasets, patient_dir+"/CT.")
                slice_orders = util_dicom.slice_order(patient_dir)
                img_data, mask_data = util_dicom.get_img_mask_voxel(slice_orders, mask_dict, patient_dir+"/CT." )

                # Create bounding box
                # todo @fruffini
                # boundind_box_body -> xlsx
                # bounding_box_lungs -> xlsx


                import matplotlib.pyplot as plt

                i = 0
                for img, mask in zip(img_data, mask_data):

                    # Clip image in -1000 2000
                    lower = -1000
                    upper = 2000
                    img = np.clip(img, lower, upper)
                    # Normalize bet
                    img = (img - lower) / (upper - lower)

                    # Check if some number diverse from 0 in the mask
                    if np.sum(mask) == 0:
                        continue

                    # Plot image
                    plt.imshow(img, cmap='gray')
                    # Save low res
                    plt.savefig(os.path.join(reports_dir,  f"img_{i}.png"))
                    i += 1
                print(i)

            except:
                print("No segmentation file found")

    print("May the force be with you")
