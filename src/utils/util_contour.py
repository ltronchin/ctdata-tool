import os
import numpy as np
import pydicom
from tqdm import tqdm
from src.utils import util_dicom
import cv2



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
    return final_mask


def get_slices_and_masks(ds_seg, roi_names=[], patient_dir=str):
    """
    This function returns the slices and the mask of a specific roi all ordered by the
    Patient Image Position inside the dicom metadata.
    :param patient_dir: path to the patient directory
    :param ds_seg: dicom dataset of the segmentation file
    :param roi_names: list of the roi names to extract
    :return: img_voxel: img_voxel: list of all the array slices,
             metadatas: ordered list of the dicom metadata,
             voxel_by_rois: dictionary of the ordered mask voxel for each roi
    """

    slice_orders = util_dicom.slice_order(patient_dir)
    # Load slices :
    img_voxel = []
    metadatas = []
    voxel_by_rois = {name: [] for name in roi_names}
    pbar = tqdm(slice_orders)
    for img_id, _ in pbar:
        pbar.set_description("Processing Patient ID:  %s" % os.path.basename(patient_dir))
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


def find_bboxes(mask):
    # get contours
    contours = cv2.findContours(mask.astype(np.int32).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    bboxes = []
    for cntr in contours:
        bbox = cv2.boundingRect(cntr)  # returns bbox = (x, y, w, h), where x,y are the coordinates of the top left
        # corner and w,h are the width and height of the bounding box
        area = bbox[2] * bbox[3]
        bboxes.append((list(bbox), area))
    if len(bboxes) > 0:
        max_bbox, left_bbox, right_bbox = find_max_left_right(bboxes) # returns the two biggest boxes and the left and right boxes
        return max_bbox, left_bbox, right_bbox
    else:
        return [0,0,0,0], [0,0,0,0], [0,0,0,0]


def find_max_left_right(bboxes):
    """
    This function returns the two biggest boxes and the left and right boxes
    :param bboxes: list of boxes
    :return: bbox_tot, bbox_left, bbox_right in the format (x, y, w, h)
    """
    # This function extract the two biggest boxes
    areas = [elem[1] for elem in bboxes]
    boxes_ = [elem[0] for elem in bboxes]
    if len(boxes_) < 2:
        bbox_tot = boxes_[0]
        return bbox_tot, None, None
    else:
        two_boxes = [boxes_[i] for i in list(np.argsort(areas, axis=0)[-2:])]
        # MAX BOX TOTAL
        matrix = np.zeros((2,len(two_boxes[0])))
        for i, bbox in enumerate(two_boxes):
            x, y, w, h = bbox
            matrix[i, 0] = int(x)
            matrix[i, 1] = int(x + w)
            matrix[i, 2] = int(y)
            matrix[i, 3] = int(y + h)
        bbox_tot = [np.min(matrix[:,0]), np.min(matrix[:,2]), np.max(matrix[:,1]) - np.min(matrix[:,0]), np.max(matrix[:,3]) - np.min(matrix[:,2])]
        bbox_left = two_boxes[np.argmin(matrix[:,0])]
        bbox_right = two_boxes[np.argmax(matrix[:,0])]
        return bbox_tot, bbox_left, bbox_right



def get_bounding_boxes(volume, z_index=2):
    """
    This function returns the bounding boxes for each planar image in a volume.
    :param volume: numpy array of the volume
    :return: dict of bounding boxes
    """
    output_bboxes = {}
    for z_i in range(volume.shape[z_index]):
        max_bbox, left_bbox, right_bbox = find_bboxes(volume[:,:,z_i])
        output_bboxes[z_i] = {'max': max_bbox, 'left': left_bbox, 'right': right_bbox}
    return output_bboxes

def get_maximum_bbox_over_slices(list_bboxes):
    return [int(np.min([bbox[0] for bbox in list_bboxes])), int(np.min([bbox[1] for bbox in list_bboxes])), int(np.max([bbox[2] for bbox in list_bboxes])), int(np.max([bbox[3] for bbox in list_bboxes]))]