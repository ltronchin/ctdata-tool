
import sys

from pandarallel import pandarallel

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])


import os
import yaml
import numpy as np
import pandas as pd
import glob
import pydicom
from src.utils import util_data, util_path, util_contour



import argparse
argparser = argparse.ArgumentParser(description='Prepare data for training')
argparser.add_argument('-c', '--config',
                       help='configuration file path', default='./configs/prepare_data2d_RC.yaml')
args = argparser.parse_args()


pandarallel.initialize()






















def save_slices(patient_dir, cfg):
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
    slices_dir = os.path.join(processed_dir, 'data')

    # Patient Information Dicom
    patients_info_file = os.path.join(interim_dir, 'patients_info.xlsx')
    patients_info_df = pd.read_excel(patients_info_file).set_index('PatientID')

    # RTstruct file

    rtstruct_file = os.path.join(interim_dir, 'structures.xlsx')

    preprocessing = cfg['data']['preprocessing']
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
            patient_dir_processed = os.path.join(slices_dir, patient_fname)
            patient_images_dir_path = os.path.join(patient_dir_processed)
            util_path.create_replace_existing_path(patient_images_dir_path, force=True, create=True)
            # Get Dicom Informations
            dicom_info_patient = patients_info_df.loc[patient_fname].to_frame()
            # Volume slices
            img_voxel, metadatas, rois_dict = util_contour.get_slices_and_masks(ds_seg, roi_names=[],
                                                                  patient_dir=patient_dir)
            # Stack all the slices
            img_voxel = np.stack(img_voxel, axis=2)
            HU_voxel = np.zeros(img_voxel.shape, dtype=np.float32)
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
                # Clip HU values
                slice_HU = slice_HU.clip(preprocessing['range']['min'], preprocessing['range']['max'])
                # Reassemble the volume
                HU_voxel[:, :, z_i] = slice_HU
            # INTERPOLATION: Interpolate the volume such to obtain a pixel spacing [1 mm , 1 mm], if the parameter
            # interpolate_z is True the target spacing is [1 mm, 1 mm, 1 mm]
            HU_int_voxel = util_data.interpolation_slices(dicom_info_patient,
                                              HU_voxel,
                                              index_z_coord=2,
                                              target_planar_spacing=[1, 1],
                                              interpolate_z=False,
                                              is_mask=False,
                                              )
            # Set padding to the minimum clipping value
            HU_int_voxel = util_data.set_padding_to_air(HU_int_voxel, padding_value=preprocessing['range']['min'], new_value=preprocessing['range']['min'])
            # Intersect pixel with the body-mask if is present in the mask dataset
            masks_file = os.path.join(processed_dir, '3D_masks', patient_fname, f'Masks_interpolated_{patient_fname}_.pkl.gz')
            bbox_file = os.path.join(processed_dir, '3D_masks', patient_fname, f'bboxes_interpolated_{patient_fname}.xlsx')
            bbox_df = pd.read_excel(bbox_file).rename(columns={'Unnamed: 0': 'ROI_id'})
            # Select only slices with the lungs inside
            selection_slices_with_lungs = bbox_df['bbox_lungs'] != '[0, 0, 0, 0]'
            # Find max bbox for body, lungs and lungs-lesions for slices with lungs inside
            max_body_bbox = util_contour.get_maximum_bbox_over_slices([eval(bbox) for bbox in bbox_df.loc[selection_slices_with_lungs, 'bbox_body'].tolist()])
            max_lungs_bbox = util_contour.get_maximum_bbox_over_slices([eval(bbox) for bbox in bbox_df.loc[selection_slices_with_lungs, 'bbox_lungs'].tolist()])
            max_ll_bbox = util_contour.get_maximum_bbox_over_slices([eval(bbox) for bbox in bbox_df.loc[selection_slices_with_lungs, 'bbox_ll'].tolist()])
            number_of_CT_slices_original = HU_int_voxel.shape[2]
            # Get masks
            masks_dictionary_dataset = util_data.load_volumes_with_names(masks_file)
            # Intersect pixel with the body-mask if is present in the mask dataset
            dictionary_masks = masks_dictionary_dataset.copy()
            if 'Body' in masks_dictionary_dataset.keys():
                Body_mask = masks_dictionary_dataset['Body']
                Volume_mask_and_original = util_contour.Volume_mask_and_original(HU_int_voxel, Body_mask)
                # Select only slices with the lungs inside
                HU_lungs_voxel = Volume_mask_and_original[:,:,selection_slices_with_lungs]
                number_of_CT_slices_lungs_only = HU_lungs_voxel.shape[2]
                # Reduce each mask volume to the exact volume of slices that contains the lungs
                dictionary_masks.pop('Body')
            dictionary_masks.pop('Lesions')
            for key, mask in dictionary_masks.items():
                print('SAVING MASK: ', key)
                mask_file_reduced = os.path.join(patient_images_dir_path,
                                                 f'{key}_mask_{patient_fname}.pkl.gz')
                util_data.save_volumes_with_names(mask[:, :, selection_slices_with_lungs], mask_file_reduced)
            ROIS_ids = bbox_df.loc[ selection_slices_with_lungs, 'ROI_id'].tolist()
            # Save single slices
            for z_index, slice_number in zip(range(HU_lungs_voxel.shape[2]), ROIS_ids):
                # Slice file creation
                slice_file = os.path.join(patient_images_dir_path, f'{slice_number}_slice.pickle')
                # Slice extraction
                slice = HU_lungs_voxel[:, :, z_index]
                # Save slice
                util_data.save_volumes_with_names(slice, slice_file)
            # Save patients informations
            info_patients_final.append(
                {
                    'ID': patient_fname,
                    '#slices_original': number_of_CT_slices_original,
                    '#slices_lungs_only': number_of_CT_slices_lungs_only,
                    'max_body_bbox': max_body_bbox,
                    'max_lungs_bbox': max_lungs_bbox,
                    'max_ll_bbox': max_ll_bbox,
                    'slices_in': ROIS_ids[0],
                    'slices_fin': ROIS_ids[-1],
                    'ROIs_names': list(dictionary_masks.keys()),
                    'CR': dicom_info_patient.loc['RC'][0],
                }
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
    # Parameters
    dataset_name = cfg['data']['dataset_name']
    img_dir = cfg['data']['img_dir']
    img_dir = os.path.join(img_dir, dataset_name)

    # List all patients in the source directory
    patients_list = glob.glob(os.path.join(img_dir, "*"))  # patient folders
    # Parallelize the elaboration of each patient
    info_patients_final = pd.Series(patients_list).parallel_apply(save_slices, cfg=cfg)

    # Save patients informations
    info_patients_final_clean = [value for value in [i for i in info_patients_final.tolist() if i is not None] if
                                 len(value) != 0]
    info_patients_final_df = pd.DataFrame([value[0] for value in info_patients_final_clean]).set_index('ID')
    final_file = './data/processed/RC/data/data.xlsx'
    info_patients_final_df.to_excel(final_file)

    print("May the force be with you")


