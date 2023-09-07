import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import os
import yaml
import numpy as np
import pydicom



from src.utils import util_datasets

import argparse
argparser = argparse.ArgumentParser(description='Prepare data for training')
argparser.add_argument('-c', '--config',
                       help='configuration file path', default='./configs/prepare_data2d_RG.yaml')
args = argparser.parse_args()

if __name__ == '__main__':

    # Config file
    print("Upload configuration file")
    with open(args.config) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Parameters

    dataset_name = cfg['data']['dataset_name']
    # Dataset Class Selector
    dataset_class_selector = {'NSCLC-RadioGenomics': util_datasets.NSCLCRadioGenomics, 'AERTS': util_datasets.AERTS, 'RC': util_datasets.RECO}


    # Initialize dataset class
    Dataset_class = dataset_class_selector[dataset_name](cfg=cfg)
    Dataset_class.load_dicom_info_report()

    # List all patient directories
    patients_list, patients_ids_list = Dataset_class.get_patients_directories()
    data = []

    for patient_dir in patients_list:

        if os.path.isdir(patient_dir):
            print("\n")
            print(f"Patient: {patient_dir}")

            # Load idpatient from dicom file

            try:
                # Select name from label file
                dicom_files, CT_scan_dir, seg_files, RTSTRUCT_dir = Dataset_class.get_dicom_files(patient_dir, segmentation_load=True)

                ds_seg = pydicom.dcmread(seg_files[0])
                if 'Amir' in patient_dir:
                    pass
                # Get structures
                structures, rois_classes = Dataset_class.get_structures_names(ds_seg).get_structures_and_classes()

                data.append({'patient_dir': patient_dir, 'structures': structures, 'rois_classes': list(np.unique(rois_classes))})

            except Exception as e:
                print(e)
    # Create Save
    Dataset_class.create_save_structures_report(data)
    Dataset_class.save_clinical()
    print("May the force be with you")
