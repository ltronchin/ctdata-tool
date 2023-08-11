from pandarallel import pandarallel
from math import sin


import re
import shutil
import sys
from copy import deepcopy

from src.utils.util_contour import get_maximum_bbox_over_slices

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

pandarallel.initialize()

def elaborate_patient_volume(patient_dir, cfg):


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


    try:
        if os.path.isdir(patient_dir):
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
            patient_folders = util_path.make_patient_folders(patient_fname, processed_dir, dataset_name,
                                                             extension='tiff', force=True, create=True)

            # Get ROIs from segmentation file
            struct_info = eval(rtstruct_df.loc[patient_dir]['structures'])

            # Get Dicom Informations
            dicom_info_patient = patients_info_df.loc[patient_fname].to_frame()

            # Create mask volume for each ROI
            img_voxel, metadatas, rois_dict = util_contour.get_slices_and_masks(ds_seg,
                                                                                roi_names=list(struct_info.values()),
                                                                                patient_dir=patient_dir)
            rois_dict_backup = rois_dict.copy()
            # Obtain Bounding Box for Lungs ROIs and Body ROI
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
                    dict_final_masks['Body'] = rois_dict[roi_name].astype(np.int8) * 255
                    continue
                else:
                    continue

                mask_roi_final = util_contour.Volume_mask_and_or(voxel_final_lungs_and_lesion > 0.5, mask_roi > 0.5,
                                                                 OR=True)
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
                                                                                            dict_final_masks[
                                                                                                'Lungs_Lesions'],
                                                                                            index_z_coord=2,
                                                                                            target_planar_spacing=[1,
                                                                                                                   1],
                                                                                            interpolate_z=False,
                                                                                            is_mask=True)
            # 4) Body
            dict_final_masks_interpolated['Body'] = util_data.interpolation_slices(dicom_info_patient,
                                                                                   dict_final_masks['Body'],
                                                                                   index_z_coord=2,
                                                                                   target_planar_spacing=[1, 1],
                                                                                   interpolate_z=False,
                                                                                   is_mask=True)

            # ----------------------- Interpolated -------------------------- #

            # GET lung/body BBOX interpolated
            dict_bounding_box_body_interpolated = util_contour.get_bounding_boxes(
                volume=dict_final_masks_interpolated['Body'])
            dict_bounding_box_lungs_interpolated = util_contour.get_bounding_boxes(
                volume=dict_final_masks_interpolated['Lungs'])
            dict_bounding_box_ll_interpolated = util_contour.get_bounding_boxes(
                volume=dict_final_masks_interpolated['Lungs_Lesions'])

            # Save bounding box report INTERPOLATED BODY
            bounding_box_interpolated_df = pd.DataFrame(dict_bounding_box_body_interpolated).T.drop(
                columns=['left', 'right']).rename(columns={'max': 'bbox_body'})
            maximum_bbox_body = get_maximum_bbox_over_slices(bounding_box_interpolated_df.loc[:, 'bbox_body'].to_list())
            bounding_box_interpolated_df.loc[:, 'max_bbox_body'] = [maximum_bbox_body for i in range(
                len(dict_bounding_box_body_interpolated))]

            # Save bounding box report INTERPOLATED LUNGS
            bounding_box_lungs_interpolated_df = pd.DataFrame(dict_bounding_box_lungs_interpolated).T.rename(
                columns={'max': 'bbox_lungs', 'right': 'right_lung', 'left': 'left_lung'})
            maximum_bbox_lungs = get_maximum_bbox_over_slices(
                [value for value in bounding_box_lungs_interpolated_df.loc[:, 'bbox_lungs'].to_list() if
                 not sum(value) == 0])
            bounding_box_lungs_interpolated_df.loc[:, 'max_bbox_lungs'] = [maximum_bbox_lungs for i in range(
                len(dict_bounding_box_lungs_interpolated))]

            # Save bounding box report INTERPOLATED LUNGS + LESIONS
            bounding_box_ll_interpolated_df = pd.DataFrame(dict_bounding_box_ll_interpolated).T.rename(
                columns={'max': 'bbox_ll', 'right': 'right_ll', 'left': 'left_ll'})
            maximum_bbox_ll = get_maximum_bbox_over_slices(
                [value for value in bounding_box_ll_interpolated_df.loc[:, 'bbox_ll'].to_list() if not sum(value) == 0])
            bounding_box_ll_interpolated_df.loc[:, 'max_bbox_ll'] = [maximum_bbox_ll for i in range(
                len(dict_bounding_box_ll_interpolated))]

            # Concatenate bounding box report INTERPOLATED for BODY and LUNGS
            df_interpolated = pd.concat(
                [bounding_box_interpolated_df, bounding_box_lungs_interpolated_df, bounding_box_ll_interpolated_df],
                axis=1)
            df_interpolated.to_excel(
                os.path.join(patient_folders['patient_path'], f'bboxes_interpolated_{patient_fname}.xlsx'))

            # ----------------------- Original -------------------------- #
            # GET lung/body BBOX original
            dict_bounding_box_body = util_contour.get_bounding_boxes(volume=dict_final_masks['Body'])
            dict_bounding_box_lungs = util_contour.get_bounding_boxes(volume=dict_final_masks['Lungs'])
            dict_bounding_box_ll = util_contour.get_bounding_boxes(volume=dict_final_masks['Lungs_Lesions'])

            # Save bounding box report BODY
            bounding_box_df = pd.DataFrame(dict_bounding_box_body).T.drop(columns=['left', 'right']).rename(
                columns={'max': 'bbox_body'})
            maximum_bbox_body = get_maximum_bbox_over_slices(bounding_box_df.loc[:, 'bbox_body'].to_list())
            bounding_box_df.loc[:, 'max_bbox_body'] = [maximum_bbox_body for i in range(len(dict_bounding_box_body))]

            # Save bounding box report LUNGS
            bounding_box_lungs_df = pd.DataFrame(dict_bounding_box_lungs).T.rename(
                columns={'max': 'bbox_lungs', 'right': 'right_lung', 'left': 'left_lung'})
            maximum_bbox_lungs = get_maximum_bbox_over_slices(
                [value for value in bounding_box_lungs_df.loc[:, 'bbox_lungs'].to_list() if not sum(value) == 0])
            bounding_box_lungs_df.loc[:, 'max_bbox_lungs'] = [maximum_bbox_lungs for i in
                                                              range(len(dict_bounding_box_lungs))]

            # Save bounding box report LUNGS + LESIONS
            bounding_box_ll_df = pd.DataFrame(dict_bounding_box_ll).T.rename(
                columns={'max': 'bbox_ll', 'right': 'right_ll', 'left': 'left_ll'})
            maximum_bbox_ll = get_maximum_bbox_over_slices(
                [value for value in bounding_box_ll_df.loc[:, 'bbox_ll'].to_list() if not sum(value) == 0])
            bounding_box_ll_df.loc[:, 'max_bbox_ll'] = [maximum_bbox_ll for i in range(len(dict_bounding_box_ll))]

            # Concatenate bounding box report for BODY and LUNGS
            df = pd.concat([bounding_box_df, bounding_box_lungs_df, bounding_box_ll_df], axis=1)
            df.to_excel(os.path.join(patient_folders['patient_path'], f'bboxes_{patient_fname}.xlsx'))

            # Save volumes
            volumes_file = os.path.join(patient_folders['masks'], f'Masks_interpolated_{patient_fname}_.pkl.gz')
            util_data.save_volumes_with_names(dict_final_masks_interpolated, volumes_file)

    except AssertionError as e:
        print(e)
        print("No segmentation file found")


if __name__ == '__main__':

    # Config file
    print("Upload configuration file")
    config_file = args.config



    with open(config_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)


    img_dir = cfg['data']['img_dir']
    dataset_name = cfg['data']['dataset_name']
    img_dir = os.path.join(img_dir, dataset_name)
    # List all patients in the source directory
    patients_list = glob.glob(os.path.join(img_dir, "*")) # patient folders

    # Parallelize the elaboration of each patient
    pd.Series(patients_list).parallel_apply(elaborate_patient_volume, cfg=cfg)


    print("May the force be with you")
