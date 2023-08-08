import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import glob
import pydicom
import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Config file
    print("Upload configuration file")
    with open('./configs/prepare_data2d_RC.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Parameters
    img_dir = cfg['data']['img_dir']
    label_file = cfg['data']['label_file']
    dataset_name = cfg['data']['dataset_name']
    interim_dir =os.path.join(cfg['data']['interim_dir'], dataset_name)
    processed_dir = os.path.join(cfg['data']['processed_dir'], dataset_name)
    reports_dir = os.path.join(cfg['reports']['reports_dir'], dataset_name)

    # List all patient directories
    patient_list = os.listdir(img_dir)

    # Prepare dictionaries to hold the statistics
    pixel_statistics = {
        'patient': [],
        'mean': [],
        'median': [],
        'min': [],
        'max': []
    }
    df_stats = pd.DataFrame(columns=['patient', 'mean', 'median', 'min', 'max'])

    for patient in tqdm(patient_list):
        pixel_statistics['patient'] = patient

        patient_dir = os.path.join(img_dir, patient)

        # Check if the current path is a directory
        if os.path.isdir(patient_dir):

            # List all .dcm files in the patient directory
            dicom_files = glob.glob(os.path.join(patient_dir, 'CT*.dcm'))
            dicom_files.sort()

            pixel_data = []
            # Read each DICOM file and append pixel data to the list
            for file in dicom_files:
                ds = pydicom.dcmread(file)
                pixel_array = ds.pixel_array
                pixel_data.extend(pixel_array.flatten())

            pixel_statistics['mean'] = np.mean(pixel_data)
            pixel_statistics['median'] = np.median(pixel_data)
            pixel_statistics['min'] = np.min(pixel_data)
            pixel_statistics['max'] = np.max(pixel_data)

            df_stats.loc[len(df_stats)] = pixel_statistics

    if not os.path.exists(interim_dir):
        os.makedirs(interim_dir)
    df_stats.to_excel(os.path.join(interim_dir, 'pixels_info.xlsx'), index=False)

    print("May the force be with you")