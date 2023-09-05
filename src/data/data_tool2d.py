
import sys

from pandarallel import pandarallel

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import nibabel as nb
import os
import yaml
import numpy as np
import pandas as pd
import glob
import pydicom
from src.utils import util_data, util_path, util_contour, util_datasets

import argparse
argparser = argparse.ArgumentParser(description='Prepare data for training')
argparser.add_argument('-c', '--config',
                       help='configuration file path', default='./configs/prepare_data2d_AERTS.yaml')
argparser.add_argument('-v', '--save_volume', help='save volume in nifti format', action='store_true')
argparser.add_argument('-i', '--interpolate_v', action='store_true')


args = argparser.parse_args()


pandarallel.initialize()

def saveCT(patient_dir, cfg):
    """
    This function elaborates the volume of a patient and
    :param patient_dir:
    :param cfg:
    :return:
    """
    # Output file for bbox, directories, labels for all the patients
    info_patients_final = list()

    dataset_name = cfg['data']['dataset_name']

    interim_dir = os.path.join(cfg['data']['interim_dir'], dataset_name)
    processed_dir = os.path.join(cfg['data']['processed_dir'], dataset_name)
    name_dict = {(True, True): 'volumes_V', (True, False): 'volumes_I', (False, False): 'slices'}
    data_directory_name = name_dict[cfg['save_volume'], cfg['interpolate_v']]
    slices_dir = os.path.join(processed_dir, data_directory_name)

    # Patient Information Dicom
    patients_info_file = os.path.join(interim_dir, 'patients_info.xlsx')
    patients_info_df = pd.read_excel(patients_info_file).set_index('PatientID')

    # RTstruct file

    preprocessing = cfg['data']['preprocessing']
    # Metadata for CT scans
    metadata_file = cfg['data']['metadata_file']
    metadata = pd.read_csv(metadata_file, sep=',') if metadata_file is not None else None
    try:
        if os.path.isdir(patient_dir):
            print("\n")
            print(f"Patient: {patient_dir}")
            final_info_patient = dict()
            # Load idpatient from dicom file
            dicom_files, CT_scan_dir, seg_files, RTSTRUCT_dir = util_datasets.get_dicom_files(dataset_name=dataset_name,
                                                                   patient_dir=patient_dir,
                                                                   metadata=metadata,
                                                                   segmentation_load=True)

            # Open files
            ds_seg = pydicom.dcmread(seg_files[0])
            ds = pydicom.dcmread(dicom_files[0])

            # Select id_patient
            patient_fname = getattr(ds, 'PatientID', None)
            assert patient_fname is not None, "Patient ID not found"
            final_info_patient['ID'] = patient_fname
            # Create patient folders
            patient_dir_processed = os.path.join(slices_dir, patient_fname)
            patient_images_dir_path = os.path.join(patient_dir_processed)
            util_path.create_replace_existing_path(patient_images_dir_path, force=True, create=True)

            # Get Dicom informations
            dicom_info_patient = patients_info_df.loc[patient_fname].to_frame()

            # Create mask volume for each ROI
            img_voxel, metadatas, rois_dict = util_contour.get_slices_and_masks(ds_seg,
                                                                                slices_dir=CT_scan_dir,
                                                                                dataset_name=dataset_name)

            # Stack all the slices
            img_voxel = np.stack(img_voxel, axis=2)
            HU_voxel = np.zeros(img_voxel.shape, dtype=np.float32)


            # Select only slices that contains the lungs, if there is the lungs mask in the dataset
            masks_file = os.path.join(interim_dir, '3D_masks', patient_fname, f'Masks_interpolated_{patient_fname}_.pkl.gz')
            bbox_file = os.path.join(interim_dir, '3D_masks', patient_fname, f'bboxes_interpolated_{patient_fname}.xlsx')
            bbox_df = pd.read_excel(bbox_file).rename(columns={'Unnamed: 0': 'ROI_id'})
            # ROIs IDs
            ROIS_ids = bbox_df.loc[:, 'ROI_id'].tolist()
            # [*0] If there are musk lungs in the dataset select the slices in the volume that contains lungs

            # Get masks
            masks_dataset = util_data.load_volumes_with_names(masks_file)
            masks_target = list(masks_dataset.keys())
            # Number of slices original
            number_of_CT_slices_original = img_voxel.shape[2]
            final_info_patient['#slices_original'] = number_of_CT_slices_original
            if 'Lungs' in masks_target:
                # Select only slices with the lungs inside
                selection_slices_with_lungs = bbox_df['bbox_lungs'] != '[0, 0, 0, 0]'

                dict_max_bbox = {
                    f'max_bbox_{mask_class.lower()}': util_contour.get_maximum_bbox_over_slices([eval(bbox) for bbox in bbox_df.loc[selection_slices_with_lungs,
                    f'bbox_{mask_class.lower()}'].tolist() if sum(eval(bbox)) != 0]) for mask_class in masks_target
                }



                # Select only slices with the lungs inside
                img_voxel = img_voxel[:, :, selection_slices_with_lungs]

                # Number of slices with lungs only
                number_of_CT_slices_lungs_only = img_voxel.shape[2]
                final_info_patient['#slices_lungs_only'] = number_of_CT_slices_lungs_only

                for mask_class in masks_target:
                    # Replace the mask volume with the reduced one
                    masks_dataset[mask_class] = masks_dataset[mask_class][:, :, selection_slices_with_lungs]
                # ROIs IDs only lungs slices
                ROIS_ids = bbox_df.loc[selection_slices_with_lungs, 'ROI_id'].tolist()

            # ROIs IDs
            final_info_patient['slices_in'] = ROIS_ids[0]
            final_info_patient['slices_fin'] = ROIS_ids[1]
            final_info_patient['ROIs_names'] = list(masks_dataset.keys())

            # ---------------------------------------- PREPROCESSING ----------------------------------------
            # [1] Rescale to HU
            # Iterate all the patient slices
            for z_i in range(img_voxel.shape[2]):
                # Get Patient Information:
                patient_info = patients_info_df.loc[patient_fname]
                # Get slice intercept
                slice_intercept, slice_slope = patient_info['RescaleIntercept'], patient_info['RescaleSlope']
                # Get slice
                slice_pixel_array = img_voxel[:, :, z_i]
                # Rescale to HU and padding air
                slice_HU = util_data.transform_to_HU(slice=slice_pixel_array,
                                           intercept=slice_intercept,
                                           slope=slice_slope,
                                           padding_value=cfg['data']['preprocessing']['range']['min'],
                                           change_value="lower",
                                           new_value=cfg['data']['preprocessing']['range']['min'])

                # [2] Clipping HU values
                # Clip HU values
                slice_HU = slice_HU.clip(preprocessing['range']['min'], preprocessing['range']['max'])
                # Reassemble the volume
                HU_voxel[:, :, z_i] = slice_HU


            # [3] INTERPOLATION: Interpolate the volume such to obtain a pixel spacing [1 mm , 1 mm], if the parameter
            # interpolate_z is True the target spacing is [1 mm, 1 mm, 1 mm]
            if cfg['interpolate_v']:
                pass
            else:
                HU_int_voxel = util_data.interpolation_slices(dicom_info_patient,
                                              HU_voxel,
                                              index_z_coord=2,
                                              target_planar_spacing=[1, 1],
                                              interpolate_z=False,
                                              is_mask=False,
                                              )

            # [4] Set padding to the minimum clipping value
            HU_int_voxel = util_data.set_padding_to_air(HU_int_voxel, padding_value=preprocessing['range']['min'], new_value=preprocessing['range']['min'])



            # [*5] If there is the body mask in the dataset intersect the volume with the body mask
            if 'Body' in masks_target:
                # Intersect pixel with the body-mask if is present in the mask dataset
                Body_mask = masks_dataset['Body']
                HU_int_voxel = util_contour.Volume_mask_and_original(HU_int_voxel, Body_mask)

                # We are not interested in the body mask anymore
                masks_target.pop('Body')

            # ---------------------------------------- SAVE ----------------------------------------
            # SAVE all the masks inside masks_target
            for mask_class in masks_target:
                # Save mask
                mask_file_reduced = os.path.join(patient_images_dir_path,
                                                 f'{mask_class.lower()}_mask.nii.gz')

                mask_volume = masks_dataset[mask_class]

                affine = np.eye(4)
                ni_img = nb.Nifti1Image(mask_volume, affine=affine)
                nb.save(ni_img, mask_file_reduced)



            util_data.save_ct_scan(**dict(index_start=ROIS_ids[0]),volume=HU_int_voxel, directory=patient_images_dir_path, save_volume=cfg['save_volume'])

            # Add patient information to the final list

            # Save patients informations
            info_patients_final.append(
                final_info_patient
            )

        return info_patients_final
    except AssertionError as e:
        print(e)



if __name__ == '__main__':
    # Config file
    print("Upload configuration file")
    config_file = args.config
    with open(config_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    img_dir = cfg['data']['img_dir']
    dataset_name = cfg['data']['dataset_name']
    img_dir = os.path.join(img_dir, dataset_name)
    label_file = cfg['data']['label_file']
    interim_dir = os.path.join(cfg['data']['interim_dir'], dataset_name)
    metadata_file = cfg['data']['metadata_file']
    cfg['save_volume'] = args.save_volume
    cfg['interpolate_v'] = args.interpolate_v

    # List all patient directories
    patient_list_accepted, patients_list, metadata = util_datasets.get_patients_directories(dataset_name, img_dir, label_file, metadata_file)

    # Parallelize the elaboration of each patient
    #info_patients_final = saveCT(patients_list[0], cfg=cfg) # DEBUG

    info_patients_final = pd.Series(patients_list).parallel_apply(saveCT, cfg=cfg)

    # Save patients informations
    info_patients_final_clean = [value for value in [i for i in info_patients_final.tolist() if i is not None] if
                                 len(value) != 0]
    info_patients_final_df = pd.DataFrame([value[0] for value in info_patients_final_clean]).set_index('ID')
    final_file = './data/processed/RC/data/data.xlsx'
    info_patients_final_df.to_excel(final_file)

    print("May the force be with you")


