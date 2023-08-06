import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import glob
import pydicom
import pandas as pd
import yaml
from tqdm import tqdm

if __name__ == '__main__':

    # Config file
    print("Upload configuration file")
    with open('./configs/prepare_data2d.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Parameters
    img_dir = cfg['data']['img_dir']
    label_file = cfg['data']['label_file']
    interim_dir = cfg['data']['interim_dir']
    #dicom_tags = cfg['data']['dicom_tags']

    # Load label file
    df_labels = pd.read_excel(label_file)
    # Select label column
    df_labels = df_labels[['Nome paziente', 'Risposta Completa (0 no, 1 si)']]

    # List all patient directories
    patient_list = os.listdir(img_dir)

    # Prepare an empty DataFrame to hold the information
    df = pd.DataFrame(columns=['patient', 'RC']+dicom_tags)

    for patient in tqdm(patient_list):
        patient_dir = os.path.join(img_dir, patient)
        # Check if the current path is a directory
        if os.path.isdir(patient_dir):

            try:
                # Select name from label file
                label = df_labels.loc[df_labels['Nome paziente'] == patient, 'Risposta Completa (0 no, 1 si)'].values[0]
            except IndexError:
                print(f'Patient {patient} not found in label file')
                break

            try:
                # List all .dcm files in the patient directory
                dicom_files = glob.glob(os.path.join(patient_dir, 'CT*.dcm'))
                dicom_files.sort()

                # Read the first DICOM file in the directory and extract the DICOM tags
                ds = pydicom.dcmread(dicom_files[0])
                data = {
                    tag: getattr(ds, tag, None) for tag in dicom_tags
                }
                # Add patient and label to data dict
                data['patient'] = patient
                data['RC'] = label

                # Append the patient and DICOM data to the DataFrame
                df.loc[len(df)] = data
            except:
                print(f'Error in patient: {patient}')

    # Save DataFrame to a CSV file
    if not os.path.exists(interim_dir):
        os.makedirs(interim_dir)
    df.to_excel(os.path.join(interim_dir, 'patients_info.xlsx'), index=False)

print("May the force be with you")