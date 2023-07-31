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

if __name__ == '__main__':

    # Config file
    print("Upload configuration file")
    with open('./configs/prepare_data.yaml') as file:
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

    df = pd.DataFrame(columns=['img_dir', 'output_nifti', 'output_res_norm', 'mean' ,' median', 'min', 'max'])

    for patient_dir in patients_list:

        if os.path.isdir(patient_dir):
            print("\n")
            print(f"Patient: {patient_dir}")

            # Convert DICOM to NIFTI 3D volume
            interim_dir_nifti = os.path.join(interim_dir, 'nifti_volumes')
            util_path.create_dir(interim_dir_nifti)

            # Load idpatient from dicom file
            dicom_files = glob.glob(os.path.join(patient_dir, 'CT*.dcm'))
            # Open files
            seg_files = glob.glob(os.path.join(patient_dir, 'RS*.dcm'))
            ds_seg = pydicom.dcmread(seg_files[0])

            # Select idpatient
            ds = pydicom.dcmread(dicom_files[0])
            patient_fname = getattr(ds, 'PatientID', None)
            assert patient_fname is not None, "Patient ID not found"

            reports_dir_nifti = os.path.join(reports_dir, 'nifti_volumes', f'{patient_fname}')
            util_path.create_dir(reports_dir_nifti)

            # Convert to nifti volume
            output_file_nifti = os.path.join(interim_dir_nifti, f"{patient_fname}.nii.gz")
            print("Output file: ", output_file_nifti)
            vol = dicom2nifti.dicom_series_to_nifti(patient_dir, output_file_nifti, reorient_nifti=True)  # The function dicom_series_to_nifti accepts the directory containing the slices to be included in .nii.gz volume reorient_nifti=False to mantain the orientation in DICOM file
            nifti_image = vol['NII']
            util_data.visualize_nifti(outdir=reports_dir_nifti, image=nifti_image.get_fdata(), opt=preprocessing)
            data_tostats = nifti_image.get_fdata()

            # Resize and normalize NIFTI images
            interim_dir_nifti_res_norm = os.path.join(interim_dir, f'nifti_volumes_{img_size}_norm')
            util_path.create_dir(interim_dir_nifti_res_norm)
            output_file_nifit_res_norm = os.path.join(interim_dir_nifti_res_norm, f"{patient_fname}.nii.gz")
            reports_dir_nifti_resize = os.path.join(reports_dir, f'nifti_volumes_{img_size}_norm', f'{patient_fname}')
            util_path.create_dir(reports_dir_nifti_resize)
            print("Output file: ", output_file_nifit_res_norm)

            # Read, resample and resize the volume
            # If we want to resize from [512 x 512 x d] to [256 256 x d] and the pixelspacing in the source resolution is [1 1 3] we have to perform a respacing operation to [1/(512/256),  1/(512/256), 3]
            nifti_image = util_data.resize(image=nifti_image, new_shape= (img_size, img_size,  nifti_image.shape[-1]), interpolation=interpolation)
            data = util_data.normalize(data=nifti_image.get_fdata(), opt=preprocessing)
            nifti_image = nib.Nifti1Image(data, affine=nifti_image.affine)
            nib.save(nifti_image, output_file_nifit_res_norm)
            util_data.visualize_nifti(outdir=reports_dir_nifti_resize, image=nifti_image.get_fdata())

            # Save to 2D images
            # todo @ltronchin
            # todo save as .pkl
            # todo save as .tiff

            df.loc[len(df)] = [patient_dir, output_file_nifti, output_file_nifit_res_norm, np.mean(data_tostats), np.median(data_tostats), np.min(data_tostats), np.max(data_tostats)]

        df.to_excel(os.path.join(interim_dir, 'patients_map.xlsx'), index=False)

    print("May the force be with you")
