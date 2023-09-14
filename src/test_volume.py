import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils import util_contour

dataset_name = 'AERTS'

dir_ = f'/Volumes/T7/data/processed/{dataset_name}/volumes_I'
dir_mask_raw = f'/Volumes/T7/data/interim/{dataset_name}/3D_masks'
maxes = pd.DataFrame(columns=['max_bbox_lungs', 'max_bbox_lesions', 'max_bbox_lesions_lungs'])
patient_dirs = [dir for dir in os.listdir(dir_) if not dir.endswith(".xlsx")]
data = pd.read_excel(os.path.join(dir_ ,'data.xlsx')).set_index('ID', drop=False)
for dir_par in tqdm(patient_dirs):

    patient_id = dir_par.split('/')[-1]
    volume_path = dir_par + '/volume.nii.gz'
    mask_dir_patient = dir_mask_raw + '/' + patient_id
    bbox_data = glob.glob(mask_dir_patient + '/*.xlsx')
    bbox_df = pd.read_excel(bbox_data[0]).drop(columns=['Unnamed: 0'])
    selection_slices_with_lungs = bbox_df['bbox_lungs'] != '[0, 0, 0, 0]'
    # PATH TO VOLUMES
    v_file = os.path.join(dir_, dir_par, 'volume.nii.gz')
    m_file = os.path.join(dir_, dir_par, 'lungs_mask.nii.gz')
    l_file = os.path.join(dir_, dir_par, 'lesions_mask.nii.gz')
    m_l_file = os.path.join(dir_, dir_par, 'lesions_lungs_mask.nii.gz')

    # LOAD VOLUMES
    nii_volume = nib.load(v_file).dataobj
    volume = np.array(nii_volume)
    mask = np.array(nib.load(m_file).dataobj)
    lesion = np.array(nib.load(l_file).dataobj)
    mask_lesion = np.array(nib.load(m_l_file).dataobj)



    # ----------------------- Max BBOX -------------------------- #
    masks_target = ['Lungs', 'Lesions', 'Lesions_Lungs']
    dict_max_bbox = {
        f'max_bbox_{mask_class.lower()}': util_contour.get_maximum_bbox_over_slices([eval(bbox) for bbox in bbox_df.loc[selection_slices_with_lungs,
        f'bbox_{mask_class.lower()}'].tolist() if sum(eval(bbox)) != 0]) for mask_class in masks_target
    }

    # ADD TO DATA:
    par_df_bbox = pd.DataFrame([dict_max_bbox], index=[patient_id])
    maxes = pd.concat([maxes, par_df_bbox])


    if volume.shape[2] == mask.shape[2]:
        continue
    else:
        volume = volume[:, :, selection_slices_with_lungs]
        affine = np.eye(4)
        ni_img = nib.Nifti1Image(volume, affine=np.eye(4))
        nib.save(ni_img, v_file)








pass



img_v = nib.load(volume_path)
img_m = nib.load(mask_path)
img_m_l = nib.load(mask_lesion_path)

pass