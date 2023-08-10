
import re
import shutil
import sys
from copy import deepcopy

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import yaml
import pandas as pd
import glob
import pydicom
from src.utils import util_path, util_data, util_contour
import numpy as np
import argparse


argparser = argparse.ArgumentParser(description='Prepare data for training')
argparser.add_argument('-c', '--config',
                       help='configuration file path', default='./configs/prepare_data2d_RC.yaml')
argparser.add_argument('-i', '--interpolate_xy', action='store_true',
                       help='debug mode', default=False)
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

        try:
            if os.path.isdir(patient_dir):
                print("\n")
                print(f"Patient: {patient_dir}")

                # Load idpatient from dicom file
                dicom_files = glob.glob(os.path.join(patient_dir, 'CT*.dcm'))
                # Open files
                seg_files = glob.glob(os.path.join(patient_dir, 'RS*.dcm'))
                assert len(seg_files) > 0, "More than one dicom file found"
                ds_seg = pydicom.dcmread(seg_files[0])

                # Select idpatient
                ds = pydicom.dcmread(dicom_files[0])
                patient_fname = getattr(ds, 'PatientID', None)
                assert patient_fname is not None, "Patient ID not found"

                # Create patient folders
                patient_folders = util_path.make_patient_folders(patient_fname, processed_dir, dataset_name, extension='tiff', force=False, create=True)

                # Get ROIs from segmentation file
                struct_info = eval(rtstruct_df.loc[patient_dir]['structures'])


                # Get Dicom Informations
                dicom_info_patient = patients_info_df.loc[patient_fname].to_frame()

                # Create mask volume for each ROI
                img_voxel, metadatas, rois_dict = util_contour.get_slices_and_masks(ds_seg, roi_names=list(struct_info.values()), patient_dir=patient_dir)
                rois_dict_backup = rois_dict.copy()
                # Obtain Bounding Box for Lungs ROIs and Body ROI
                print('Obtaining bounding box for lungs and body')
                rois_dict = rois_dict_backup.copy()

                # Elaborate masks
                dict_final_masks = {}
                voxel_final_lungs_and_lesion = np.zeros(
                    shape=(img_voxel[0].shape[0], img_voxel[0].shape[1], len(img_voxel)))
                voxel_final_lungs = np.zeros(
                    shape=(img_voxel[0].shape[0], img_voxel[0].shape[1], len(img_voxel)))
                voxel_final_lesions = np.zeros(
                    shape=(img_voxel[0].shape[0], img_voxel[0].shape[1], len(img_voxel)))

                for roi_name, roi_mask in rois_dict.items():
                    pattern = re.compile('(^lung[0-9._\s])', re.IGNORECASE)
                    if "polmone " in roi_name.lower() or pattern.search(roi_name):
                        mask_roi = np.stack(rois_dict[roi_name], axis=2) > 0.5
                        mask_roi_lungs = util_contour.Volume_mask_and_or(voxel_final_lungs > 0.5, mask_roi, OR=True)
                        rois_dict[roi_name] = mask_roi
                        voxel_final_lungs = deepcopy(mask_roi_lungs)


                    elif re.search("^CTV[0-9]{0,1}", roi_name.upper()) is not None:
                        mask_roi = np.stack(rois_dict[roi_name], axis=2) > 0.5
                        rois_dict[roi_name] = mask_roi
                        mask_roi_lesions = util_contour.Volume_mask_and_or(voxel_final_lesions > 0.5, mask_roi, OR=True)
                        voxel_final_lesions = deepcopy(mask_roi_lesions)
                    elif "corpo" in roi_name.lower() or "body" in roi_name.lower() or "external" in roi_name.lower():
                        rois_dict[roi_name] = np.stack(rois_dict[roi_name], axis=2) > 0.5
                        dict_final_masks['Body'] = rois_dict[roi_name].astype(np.int8 ) * 255
                        continue
                    else:
                        continue

                    mask_roi_final = util_contour.Volume_mask_and_or(voxel_final_lungs_and_lesion > 0.5, mask_roi > 0.5, OR=True)
                    voxel_final_lungs_and_lesion = deepcopy(mask_roi_final)

                """
                create_gif(mask_roi_final, 'lung_and_lesion')
                create_gif(mask_roi_lesions, 'lesions')
                create_gif(mask_roi_lungs, 'lungs')
                """

                # Creating the final masks dictionary for all the ROIs
                voxel_final_lungs, voxel_final_lesions, voxel_final_lungs_and_lesion = voxel_final_lungs > 0.5, voxel_final_lesions > 0.5, voxel_final_lungs_and_lesion > 0.5
                dict_final_masks['Lungs'] = voxel_final_lungs.astype(np.int8) * 255
                dict_final_masks['Lesions'] = voxel_final_lesions.astype(np.int8) * 255
                dict_final_masks['Lungs_Lesions'] = voxel_final_lungs_and_lesion.astype(np.int8) * 255


                # Interpolate Masks:
                dict_final_masks_interpolated = {}
                # 1) Lungs
                dict_final_masks_interpolated['Lungs'] = util_data.interpolation_slices(dicom_info_patient,
                                                                           dict_final_masks['Lungs'],
                                                                           index_z_coord=2,
                                                                           target_planar_spacing=[1, 1],
                                                                           interpolate_z=False,
                                                                           is_mask=True)
                # 2) Lesions
                dict_final_masks_interpolated['Lesions'] = util_data.interpolation_slices(dicom_info_patient,
                                                                             dict_final_masks['Lesions'],
                                                                             index_z_coord=2,
                                                                             target_planar_spacing=[1, 1],
                                                                             interpolate_z=False,
                                                                             is_mask=True)
                # 3) Lungs_Lesions
                dict_final_masks_interpolated['Lungs_Lesions'] = util_data.interpolation_slices(dicom_info_patient,
                                                                                   dict_final_masks['Lungs_Lesions'],
                                                                                   index_z_coord=2,
                                                                                   target_planar_spacing=[1, 1],
                                                                                   interpolate_z=False,
                                                                                   is_mask=True)
                # 4) Body
                dict_final_masks_interpolated['Body'] = util_data.interpolation_slices(dicom_info_patient,
                                                                          dict_final_masks['Body'],
                                                                          index_z_coord=2,
                                                                          target_planar_spacing=[1, 1],
                                                                          interpolate_z=False,
                                                                          is_mask=True)


                # Save volumes
                volumes_file = os.path.join(patient_folders['mask'], f'Masks_{patient_fname}_.pkl.gz')
                util_data.save_volumes_with_names(dict_final_masks, volumes_file)



                # Create gif




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
        except AssertionError as e:
            print(e)
            print("No segmentation file found")



    print("May the force be with you")
