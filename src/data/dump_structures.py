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
    with open('./configs/prepare_data2d_RC.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Parameters
    dataset_name = cfg['data']['dataset_name']
    img_dir = cfg['data']['img_dir']
    label_file = cfg['data']['label_file']

    interim_dir =os.path.join(cfg['data']['interim_dir'], dataset_name)
    processed_dir = os.path.join(cfg['data']['processed_dir'], dataset_name)
    reports_dir = os.path.join(cfg['reports']['reports_dir'], dataset_name)
    img_dir = os.path.join(img_dir, dataset_name)




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
            # Create destination directory if not exist
            # TODO interim_dir_nifti = os.path.join(interim_dir, 'nifti_volumes')
            #  util_path.create_dir(interim_dir_nifti), this part is not necessary?

            # Load idpatient from dicom file
            dicom_files = glob.glob(os.path.join(patient_dir, 'CT*.dcm'))
            ds = pydicom.dcmread(dicom_files[0])
            # Open files
            seg_files = glob.glob(os.path.join(patient_dir, 'RS*.dcm'))
            try:
                # https://github.com/pydicom/pydicom/issues/961
                ds_seg = pydicom.dcmread(seg_files[0])

                # Available structures
                structures = {}
                for item in ds_seg.StructureSetROISequence:
                    name = item.ROIName
                    if "polmo" in name.lower():
                        structures[item.ROINumber] = item.ROIName
                    elif "lung" in name.lower():
                        structures[item.ROINumber] = item.ROIName
                    elif "corpo" in name.lower():
                        structures[item.ROINumber] = item.ROIName
                    elif "body" in name.lower():
                        structures[item.ROINumber] = item.ROIName
                    elif "ctv" in name.lower():
                        structures[item.ROINumber] = item.ROIName
                    else:
                        continue
                if len(structures) == 0:
                    print("No structures found")
                else:
                    print("Available structures: ", structures)
                # Add structures to dataframe
                data.append({'patient_dir': patient_dir, 'structures': structures})

            except:
                print("No segmentation file found")
    df = pd.DataFrame(data)
    if not os.path.exists(interim_dir):
        os.makedirs(interim_dir)

    df.to_excel(os.path.join(interim_dir, 'structures.xlsx'), index=False)

    print("May the force be with you")
