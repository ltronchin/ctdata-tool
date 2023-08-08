import shutil
import sys

from tqdm import tqdm

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import yaml
import numpy as np
import pandas as pd
import glob
import pydicom
import dicom2nifti
import nibabel as nib

from src.utils import util_data, util_dicom
from src.utils import util_path


#### TODO: TEMPORARY FUNCTIONS ####
# TODO: move to utils

def set_padding_to_air(image, padding_value=-1000, change_value="lower", new_value=-1000):
    """
    This function sets the padding (contour of the dicom image) to the padding value. It trims all the values below the
    padding value and sets them to this value.

    Parameters
    ----------
    image: numpy.array
        Image to modify.
    padding_value: scalar number
        Value to use as threshold in the trim process and to be set in those points.
    change_value: string, says if the values to be changed are greater or lower than the padding value.
    new_value: scalar number, value to be set in the points where the change_value is True.

    Returns
    -------
    image: numpy.array
        Modified image.

    """

    trim_map = image < padding_value
    options = {"greater": ~trim_map, "lower": trim_map}
    image[options[change_value]] = new_value

    return image

def do_nothing( *args, return_first=False, **kwargs ):
    """
    This function does nothing to the positional arguments.
    Based on the 'return_first' parameter, it returns all the inputs or just the first one.

    Parameters
    ----------
    args:
        Positional arguments.
    return_first: bool, default False
        A boolean value to determine if the function returns all its positional inputs or just the first one.
    kwargs:
        Keyword arguments added for compatibility.

    Returns
    -------
    Tuple of inputs if 'return_first' == False else the first input.

    """
    return_options = { False: args, True: args[0] }
    return return_options[ return_first ]


def replace_existing_path(path, force=False, create=True, **kwargs):
    """
    This function, based on the 'force' parameter, deletes an existing path.
    Then, based on the 'create' parameter, if the path points to a folder, it creates a new one.

    Parameters
    ----------
    path: string
        Path of the file/folder to replace.
    force: bool, default False
        A boolean value to erase or not an existing path.
    create: bool, default True
        A boolean value to create a new folder when deleted.
    kwargs:
        Keyword arguments added for compatibility.

    Returns
    -------
        None

    """

    # CHECK IF THE PATH EXISTS
    exists = os.path.exists(path)

    _, file_name = os.path.split(path)

    is_file = "." in file_name
    if exists and not is_file:
        if is_file and exists:
            os.remove(path)
        else:
            if force or exists:
                shutil.rmtree(path)
                os.makedirs(path)
            else:
                os.makedirs(path)
    else:
        os.makedirs(path)




def make_patient_folders(patient_ID, save_path, dataset_name, extension='tiff', force=False, create=True):
    """
    This function creates the folders where to save the info of a single patient.

    Parameters
    ----------
    patient_ID: string
        ID of the patient.
    save_path: string
        Path of the folder where to save the files.
    dataset_name: string
        Name of the database (e.g., CLARO_prospettico, CLARO_retrospettivo)
    force: bool, default False
        A boolean value to define whether to delete or not an existing folder.
    create: bool, default True
        A boolean value to define whether to create or not a folder not existing.
    type_of_rois: string
        Type of the rois to save (e.g., liver, lesion, etc.).

    Returns
    -------
    paths_dict: dict
        Dict of paths to the folders of the patient.

    """
    # Create the patient directory
    patient_path = os.path.join(save_path, extension, patient_ID)
    replace_existing_path(patient_path, force=force, create=create)

    # Create the CT slices directory
    patient_images_dir_path = os.path.join(patient_path, "CT_slices")
    replace_existing_path(patient_images_dir_path, force=force, create=create)

    # Create the Roi mask directory
    # todo patient_masks_dir_path = os.path.join(patient_path, f"{type_of_rois}_masks")
    #  replace_existing_path(patient_masks_dir_path, force=force, create=create)

    return {'patient_path': patient_path, 'image': patient_images_dir_path}




def transform_to_HU(slice, intercept, slope, padding_value=-1000, change_value="lower", new_value=-1000):
    """
    This function transforms to Hounsfield units all the images passed.

    Parameters
    ----------
    slice: numpy.array
        List of metadatas of the slices where to gather the information needed in the transformation.
    intercept: scalar number
        Intercept of the slice.
    slope: scalar number
        Slope of the slice.
    padding_value: scalar number
        Value to use as threshold in the trim process and to be set in those points.
    change_value: string, says if the values to be changed are greater or lower than the padding value.
    new_value: scalar number, value to be set in the points where the change_value is True.

    Returns
    -------
    images: numpy array
        transformed slice in HU with the padding set to the padding value.
    """
    intercept = np.float32(intercept)
    slope = np.float32(slope)
    slice = slice.astype("float32")

    if slope != 1:

        slice = slope * slice.astype("float32")
        slice = slice.astype("float32")
    slice += np.float32(intercept)
    # TODO qui è necessario fare il riaggiustamento per i pazienti con l'intercept sballata @ltrochin
    # Padding to air all the values below
    slice_HU_padded = set_padding_to_air(image=slice,
                                      padding_value=padding_value,
                                      change_value=change_value,
                                      new_value=new_value)
    return slice_HU_padded



import argparse
argparser = argparse.ArgumentParser(description='Prepare data for training')
argparser.add_argument('-c', '--config',
                       help='configuration file path', default='./configs/prepare_data2d_RC.yaml')
args = argparser.parse_args()


def get_slices_and_mask(ds_seg, roi_names=[]):
    """
    This function returns the slices and the mask of a specific roi.
    :param ds_seg:
    :param roi_names:
    :return:
    """

    slice_orders = util_dicom.slice_order(patient_dir)
    # Load slices :
    img_voxel = []
    metadatas = []
    for img_id, _ in tqdm(slice_orders):
        # Load the image dcm
        dcm_ = pydicom.dcmread(patient_dir + "/CT." + img_id + ".dcm")
        metadatas.append(dcm_)
        # Get the image array
        img_array = pydicom.dcmread(patient_dir + "/CT." + img_id + ".dcm").pixel_array.astype(np.float32)
        img_voxel.append(img_array)
        voxel_by_rois = {name: [] for name in roi_names}
        for roi_name in roi_names:
            mask_voxel = []
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

        if os.path.isdir(patient_dir):
            print("\n")
            print(f"Patient: {patient_dir}")

            # Load idpatient from dicom file
            dicom_files = glob.glob(os.path.join(patient_dir, 'CT*.dcm'))
            # Open files
            seg_files = glob.glob(os.path.join(patient_dir, 'RS*.dcm'))
            ds_seg = pydicom.dcmread(seg_files[0])

            # Select idpatient
            ds = pydicom.dcmread(dicom_files[0])
            patient_fname = getattr(ds, 'PatientID', None)
            assert patient_fname is not None, "Patient ID not found"

            # Create patient folders

            patient_folders = make_patient_folders(patient_fname, processed_dir, dataset_name, extension='tiff', force=False, create=True)

            # Get ROIs from segmentation file
            struct_info = eval(rtstruct_df.loc[patient_dir]['structures'])

            # filter ROIS:
            struct_info = filter_rois(struct_info, cfg['data']['rois_to_filter'])x

            img_voxel, metadatas, rois_dict = get_slices_and_mask(ds_seg, roi_names=list(struct_info.values()))




            # Iterate all the patient slices
            for dicom_file_path in dicom_files:


                # Read dicom file
                slice_dicom = pydicom.dcmread(dicom_file_path)


                # Get slice number
                slice_number = slice_dicom.InstanceNumber

                # Get Patient Information:
                patient_info = patients_info_df.loc[patient_fname]

                # Get slice intercept
                slice_intercept, slice_slope = patient_info['RescaleIntercept'], patient_info['RescaleSlope']

                # Pixel array
                slice_pixel_array = slice_dicom.pixel_array

                # Rescale to HU and padding air
                slice_HU = transform_to_HU(slice=slice_pixel_array,
                                           intercept=slice_intercept,
                                           slope=slice_slope,
                                           padding_value=cfg['data']['preprocessing']['range']['min'],
                                           change_value="lower",
                                           new_value=cfg['data']['preprocessing']['range']['min'])

                min_slice_pixel, max_slice_pixel = np.min(slice_HU), np.max(slice_HU)
                # CLIPPING: Clip the values to the range [min, max]




                # Get slice thickness
                slice_thickness = slice_dicom.SliceThickness

                # Get slice location
                slice_location = slice_dicom.SliceLocation

                # Get slice position
                slice_position = slice_dicom.ImagePositionPatient

                # Get slice orientation
                slice_orientation = slice_dicom.ImageOrientationPatient

                # Get slice pixel spacing
                slice_pixel_spacing = slice_dicom.PixelSpacing






                pass





            # Adapt HU
            # todo @ltronchin


            # Adaptive pixel spacing interpolation2d

            # todo @fruffini




            # Resize

            # Normalization
            # todo clip hu
            # todo normalize in 0,1
            # todo @ltronchin

            # Save 2D/3D volume
            # todo @fruffini

    print("May the force be with you")
