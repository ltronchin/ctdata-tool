import glob
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

from nilearn.image import reorder_img, new_img_like
import dicom2nifti
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt

from src.utils import util_path
from src.utils import util_sitk

def visualize_nifti(outdir, image, opt=None):

    print("Image shape: ", image.shape)
    print("Image min: ", np.min(image))
    print("Image max: ", np.max(image))


    for i in range(image.shape[-1]):
        img = image[:, :, i]

        if opt is not None:
            # Upper value
            lower = opt['range']['min']
            upper = opt['range']['max']
            img = np.clip(img, lower, upper)
            img = (img - lower) / (upper - lower)

        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        fig.set_tight_layout(True)
        fig.savefig(os.path.join(outdir, f"slice_{i}.jpg"), dpi=50)
        plt.close()

# ----------------------------------------------------------------------------------------------------------------------
def resize(image, new_shape, interpolation="linear"):

    #image = reorder_img(image, resample=interpolation) #Returns an image with the affine diagonal (by permuting axes).
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level) #  image.header.get_zooms() -> get the original spacing
    new_data = util_sitk.resample_to_spacing(image.get_fdata(), image.header.get_zooms(), new_spacing, interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += util_sitk.calculate_origin_offset(new_spacing, image.header.get_zooms())

    return new_img_like(image, new_data, affine=new_affine)

def normalize(data, opt):
    """follow this https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py"""
    #print("Data min: ", data.min())
    #print("Data max: ", data.max())

    # Upper value
    if opt['upper_percentile'] is not None:
        upper = np.percentile(data, opt['upper_percentile'])
    elif opt['range']['max'] is not None:
        upper = opt['range']['max']
    else:
        upper = data.max()

    # Lower value
    if opt['lower_percentile'] is not None:
        lower = np.percentile(data, opt['lower_percentile'])
    elif opt['range']['min'] is not None:
        lower = opt['range']['min']
    else:
        lower = data.min()

    data = np.clip(data, lower, upper)
    data = (data - lower) / (upper - lower)  # map between 0 and 1
    if opt['to_255']:
        data = data * 255  # put between 0 and 255

    # to float32
    data = data.astype(float)

    return data

# ----------------------------------------------------------------------------------------------------------------------

def iter_volumes(source_dir):

    patients_list = glob.glob(os.path.join(source_dir, "*"))
    def iterate_images():
        for idx, patient_dir in enumerate(patients_list):

            patient_name = util_path.get_filename_without_extension(patient_dir)
            patient_name = patient_name.split(".")[0]

            fdata = nib.load(patient_dir).get_fdata()
            depth = fdata.shape[-1] # Save the number of Slices/Channel

            for d in range(depth):
                img = fdata[:, :, d]  # based on depth index (d): 0000->0128
                if transpose_img:  # fix orientation
                    img = np.transpose(img, [1, 0])
                yield dict(img=img, name=f"{patient_name:s}_{d:05d}", folder_name=f"{patient_name:s}", depth_index=d, total_depth=depth)

            if idx >= len(patients_list) - 1:
                break

    return len(patients_list), iterate_images()

def to_file(source: str, dest: str, file_format, is_visualize=True):


    num_files, input_iter = open_image_folder_patients(source)  # Core function.

    for idx, image in enumerate(input_iter):

        img = image["img"]
        img_fname = image["name"]
        folder_name = image["folder_name"]
        archive_fname = f"{folder_name}/{img_fname}.tiff"

        util_path.create_dir(os.path.join(dest, f"{folder_name}"))

        dest_path = os.path.join(dest, archive_fname)

        if random.uniform(0, 1) > 0.1:
            if is_visualize:
                visualize(img)

