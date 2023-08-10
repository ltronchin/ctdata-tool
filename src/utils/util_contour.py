import os
import numpy as np
import pydicom
from tqdm import tqdm
from src.utils import util_dicom




def Volume_mask_and_or(volume_one, volume_two, OR=True):
    """
    This function returns the mask of the union or the intersection of two volumes.

        :param volume_one: numpy array of the first volume
        :param volume_two: numpy array of the second volume
        :param OR: boolean, if True, the function returns the union of the two volumes, if False, the function returns the intersection of the two volumes

        :return: numpy array of the mask of the union or the intersection of the two volumes
    """
    shapes = volume_one.shape
    final_mask = np.zeros(shapes)
    function_booleans = {True: np.bitwise_or, False: np.bitwise_and}
    for k in range(shapes[2]):
        for j in range(shapes[1]):
            final_mask[:, j, k] = function_booleans[OR](volume_one[:, j, k], volume_two[:, j, k])
    print('here')
    return final_mask


def get_slices_and_masks(ds_seg, roi_names=[], patient_dir=str):
    """
    This function returns the slices and the mask of a specific roi.
    :param patient_dir:
    :param ds_seg:
    :param roi_names:
    :return:
    """

    slice_orders = util_dicom.slice_order(patient_dir)
    # Load slices :
    img_voxel = []
    metadatas = []
    voxel_by_rois = {name: [] for name in roi_names}
    for img_id, _ in tqdm(slice_orders):
        # Load the image dcm
        dcm_ = pydicom.dcmread(patient_dir + "/CT." + img_id + ".dcm")
        metadatas.append(dcm_)
        # Get the image array
        img_array = pydicom.dcmread(patient_dir + "/CT." + img_id + ".dcm").pixel_array.astype(np.float32)
        img_voxel.append(img_array)

        for roi_name in roi_names:
            idx = np.where(np.array(util_dicom.get_roi_names(ds_seg)) == roi_name)[0][0]
            contour_datasets = util_dicom.get_roi_contour_ds(ds_seg, idx)
            mask_dict = util_dicom.get_mask_dict(contour_datasets, patient_dir + "/CT.")

            if img_id in mask_dict:
                mask_array = mask_dict[img_id]
            else:
                mask_array = np.zeros_like(img_array)
            voxel_by_rois[roi_name].append(mask_array)
    return img_voxel, metadatas, voxel_by_rois



def filter_rois(ROIS_dict):
    """
    Select only the rois that contains lungs, CTV and BODY.



    :param ROIS_dict:
    :return:
    """




    pass

