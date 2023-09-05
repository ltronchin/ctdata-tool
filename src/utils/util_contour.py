import os
import numpy as np
import pydicom
from scipy import ndimage
from tqdm import tqdm
from src.utils import util_dicom
import cv2


def Volume_mask_and_original(volume_original, volume_mask, fill=-1000):
    """
    This function returns the mask of the union or the intersection of two volumes: original and mask.
    :param volume_original: numpy array of the original volume
    :param volume_mask: numpy array of the mask volume
    :param fill: value to fill the empty slice
    :return: numpy array of the mask of and intersection of the two volumes
    """
    volume_out = np.zeros_like(volume_original)
    for k in range(volume_original.shape[2]):

        slice_or = volume_original[:, :, k]
        slice_mask = volume_mask[:, :, k]
        slice_mask = np.array(slice_mask, dtype=bool)
        # AND between the slice and the mask
        slice_or_and_mask = slice_or * slice_mask
        # Empty slice filled by minimum value of original volume
        empty_slice = np.ones_like(slice_or) * fill
        # Negative slice of the mask
        negative_slice_or_and_empty = (~slice_mask) * empty_slice
        # Union between the AND slice and the negative slice
        negative_slice_mask = slice_or_and_mask + negative_slice_or_and_empty
        volume_out[:, :, k] = negative_slice_mask
    return volume_out




def Volume_mask_and_or_mask(mask_one, mask_two, OR=True):
    """
    This function returns the mask of the union or the intersection of two volumes mask.

        :param mask_one: numpy array of the first volume
        :param mask_two: numpy array of the second volume
        :param OR: boolean, if True, the function returns the union of the two volumes, if False, the function returns the intersection of the two volumes

        :return: numpy array of the mask of the union or the intersection of the two volumes
    """
    shapes = mask_one.shape
    final_mask = np.zeros(shapes)
    function_booleans = {True: np.bitwise_or, False: np.bitwise_and}
    for k in range(shapes[2]):
        for j in range(shapes[1]):
            final_mask[:, j, k] = function_booleans[OR](mask_one[:, j, k], mask_two[:, j, k])
    return final_mask


def get_slices_and_masks(ds_seg, roi_names=[], slices_dir=str, dataset=None):
    """
    This function returns the slices and the mask of a specific roi all ordered by the
    Patient Image Position inside the dicom metadata.
    :param slices_dir: path to the patient directory
    :param ds_seg: dicom dataset of the segmentation file
    :param roi_names: list of the roi names to extract, if roi_names is empty, no roi will be extracted
    :param dataset_name: name of the dataset
    :return: img_voxel: img_voxel: list of all the array slices,
             metadatas: ordered list of the dicom metadata,
             voxel_by_rois: dictionary of the ordered mask voxel for each roi
    """

    slice_orders = util_dicom.slice_order(slices_dir, dataset)
    # Load slices :
    img_voxel = []
    metadatas = []
    voxel_by_rois = {name: [] for name in roi_names}
    pbar = tqdm(slice_orders)




    for img_id, _ in pbar:
        pbar.set_description("Processing Patient ID:  %s" % os.path.basename(slices_dir.split('/')[5]))
        # Load the image dcm
        slice_file = dataset.get_slice_file(slices_dir, img_id)
        dcm_ = pydicom.dcmread(slice_file)
        metadatas.append(dcm_)
        # Get the image array
        img_array = dcm_.pixel_array.astype(np.float32)
        img_voxel.append(img_array)
        dataset.set_shape(img_array.shape)
        # Get voxel-by-Rois dictionary
        voxel_by_rois = dataset.create_voxels_by_rois(ds_seg, roi_names, slices_dir, img_id)

        """        if dataset_name != 'NSCLC-RadioGenomics':
            for roi_name in roi_names:
                # GET ROIS
                idx = np.where(np.array(util_dicom.get_roi_names(ds_seg)) == roi_name)[0][0]
                contour_datasets = util_dicom.get_roi_contour_ds(ds_seg, idx)
                mask_dict = util_dicom.get_mask_dict(contour_datasets, slices_dir, img_id=img_id, dataset_name=dataset_name)
                if img_id in mask_dict:
                    mask_array = mask_dict[img_id]
                else:
                    mask_array = np.zeros_like(img_array)
                voxel_by_rois[roi_name].append(mask_array)

    if dataset_name == 'NSCLC-RadioGenomics':
        for roi_name in roi_names:
            seg = ds_seg.pixel_array
            # reorient the seg array
            seg = np.fliplr(seg.T)
            if seg.shape[2] != len(img_voxel):
                s = ds_seg.ReferencedSeriesSequence._list
                RefSOPInstanceUID_list = [slice.ReferencedSOPInstanceUID for slice in s[0].ReferencedInstanceSequence._list]
                true_slices = [img_id[0] in RefSOPInstanceUID_list for img_id, UID in zip(slice_orders, RefSOPInstanceUID_list) ]
                new_voxel = {}
                i = 0

                for img_id in slice_orders:
                    found_ = False
                    for UID in RefSOPInstanceUID_list:
                        if str(img_id[0][0]) == (UID):
                            new_voxel[img_id[0][1]] = seg[:,:,i]
                            i =+ 1
                            found_ = True
                        else:
                            pass
                    if not found_:
                        new_voxel['None' + img_id[0][1]] = np.zeros(shape=(512,512))

                seg = np.stack(list(new_voxel.values()), 2)

            voxel_by_rois[roi_name].append(seg)"""

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