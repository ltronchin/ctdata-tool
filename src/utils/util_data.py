import glob
import os
import pickle
import shutil
from copy import deepcopy
import imageio
from scipy import interpolate
import random
from nilearn.image import new_img_like
import nibabel as nib
import matplotlib.pyplot as plt
from src.utils import util_path
from src.utils import util_sitk
from src.utils.util_dicom import set_padding_to_air
import struct
import numpy as np
import zipfile

def write_idx_file(volumes_dict, volumes_file):
    num_volumes = len(volumes_dict)
    volume_size = None
    volumes = []

    for volume in volumes_dict.values():
        if volume_size is None:
            volume_size = volume.shape

        volumes.append(volume)

    with open(volumes_file, "wb") as f_volumes:
        depth, height, width = volume_size

        # Write volumes header
        f_volumes.write(struct.pack(">I", 0x00000803))  # Magic number for volumes
        f_volumes.write(struct.pack(">I", num_volumes))
        f_volumes.write(struct.pack(">I", depth))
        f_volumes.write(struct.pack(">I", height))
        f_volumes.write(struct.pack(">I", width))

        # Write volumes data
        for volume in volumes:
            for depth_slice in volume:
                for row in depth_slice:
                    f_volumes.write(row.tobytes())




def read_idx_dict(file_path):
    with open(file_path, "rb") as f:
        magic_number = struct.unpack(">I", f.read(4))[0]
        num_volumes = struct.unpack(">I", f.read(4))[0]

        if magic_number == 0x00000803:  # Magic number for volumes
            depth = struct.unpack(">I", f.read(4))[0]
            height = struct.unpack(">I", f.read(4))[0]
            width = struct.unpack(">I", f.read(4))[0]

            volumes_dict = {}

            for _ in range(num_volumes):
                volume_data = np.frombuffer(f.read(depth * height * width), dtype=np.uint8)
                volume_data = volume_data.reshape((depth, height, width))
                volumes_dict[f"volume{_ + 1}"] = volume_data

            return volumes_dict
        else:
            print("Invalid IDX file magic number.")
            return None



def save_single_npy_volume(volumes_dict, directory):
    for name, volume in volumes_dict.items():
        volume_dir = os.path.join(directory, name)
        if os.path.exists(volume_dir):
            print(f"Directory {volume_dir} already exists. Removing...")
            shutil.rmtree(volume_dir)
        else:
            os.makedirs(volume_dir, exist_ok=True)
        volume_path = os.path.join(volume_dir, f"{name}.npy")
        np.save(volume_path, volume)

def load_single_npy_volume(directory):
    volumes_dict = {}
    for name in os.listdir(directory):
        volume_dir = os.path.join(directory, name)
        if os.path.isdir(volume_dir):
            volume_path = os.path.join(volume_dir, f"{name}.npy")
            if os.path.exists(volume_path):
                volume = np.load(volume_path)
                volumes_dict[name] = volume
    return volumes_dict


def save_volumes_with_names(volumes_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(volumes_dict, f)


def load_volumes_with_names(file_path):
    with open(file_path, 'rb') as f:
        volumes_dict = pickle.load(f)
    return volumes_dict



def interpolate_slice_2D(metadata, single_slice, index_z_coord=2, target_planar_spacing=[1, 1]):
    """
    This function interpolates a slice of a patient, given its metadata and the index of the z coordinate, in order to
    obtain a pixel spacing = 1 along x and y.

    Parameters
    ----------
    metadata: dict
        Dictionary of the metadata of a dcm file.
    single_slice: np.array
        CT image.
    index_z_coord: int, default 2
        Index of the coordinate of the z axis.

    Returns
    -------
    single_slice: np.array
        Interpolated CT image.

    """
    n_rows = single_slice.shape[0]
    n_columns = single_slice.shape[1]

    PixelSpacing = eval(metadata.loc['PixelSpacing'].to_list()[0])
    x_spacing = PixelSpacing[0]
    y_spacing = PixelSpacing[1]

    patient_coords = deepcopy(eval(metadata.loc['ImagePositionPatient'].to_list()[0]))
    if len(patient_coords) > 2:
        del patient_coords[index_z_coord]

    x0, y0 = patient_coords[0], patient_coords[1]

    if abs(x0 - y0) > 1e-1:
        y0 = x0

    x = np.arange(0, n_rows * x_spacing, x_spacing)
    y = np.arange(0, n_columns * y_spacing, y_spacing)

    x, y = x + (n_rows / 2 + x0), y + (n_columns / 2 + y0)

    xnew = np.arange(0, n_rows, target_planar_spacing[0])
    ynew = np.arange(0, n_columns, target_planar_spacing[1])

    f = interpolate.interp2d(x, y, single_slice, kind="quintic", fill_value=-1000)
    interpolated_slice = f(xnew, ynew)


    return interpolated_slice


def interpolation_slices(patient_dcm_info, volume, index_z_coord=2, target_planar_spacing=[1, 1], interpolate_z=False,  z_spacing=1, is_mask=False):
    """
    This function interpolates the slices of a patient.

    Parameters
    ----------
    metadatas: list metadata
        List of the metadata of the dcm files of a patient.
    volume: np.array
        CT volume, boolean or .
    index_z_coord: int, default 2
    """

    # Output volume Mask
    volume_output = np.zeros(volume.shape)

    for z_i in range(volume.shape[index_z_coord]):
        volume_output[:, :, z_i] = interpolate_slice_2D(metadata=patient_dcm_info,
                                                        single_slice=volume[:, :, z_i],
                                                        index_z_coord=index_z_coord,
                                                        target_planar_spacing=target_planar_spacing
                                                        )
        if is_mask:
            volume_output[:, :, z_i] = volume_output[:, :, z_i] > 122.5
            volume_output[:, :, z_i] = volume_output[:, :, z_i].astype(np.int8) * 255

    # TODO OPTION TO INTERPOLATE Z
    # Set padding to air
    if not is_mask:
        set_padding_to_air(volume_output, padding_value=-1000, change_value="lower", new_value=-1000)
    else:
        return volume_output.astype(np.uint8)



def create_gif(volume, save_file):
    # Assume your CT volume is stored as a 3D NumPy array called 'volume'
    # The dimensions are (depth, height, width)

    # Convert the volume to a list of 2D slices
    slices = [volume[:, :, i] for i in range(volume.shape[2])]

    # Create a list to store the individual frames
    frames = []

    # Iterate over the slices and convert them to RGB images (for visualization purposes)
    for slice in slices:
        # Normalize the slice to the range [0, 255]
        # slice_normalized = ((slice - np.min(slice)) / (np.max(slice) - np.min(slice))) * 255
        slice = slice > 0.5
        # Convert the 2D slice to an RGB image (grayscale)
        slice_rgb = np.repeat(slice[:, :, np.newaxis], 3, axis=2).astype(int) * 255

        # Append the RGB image to the frames list
        frames.append(slice_rgb.astype(np.uint8))

    # Save the frames as a GIF file

    imageio.mimsave('./figures/ct_volume_{}.gif'.format(save_file), frames, duration=10)


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
    # image = reorder_img(image, resample=interpolation) #Returns an image with the affine diagonal (by permuting axes).
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(),
                            zoom_level)  # image.header.get_zooms() -> get the original spacing
    new_data = util_sitk.resample_to_spacing(image.get_fdata(), image.header.get_zooms(), new_spacing,
                                             interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += util_sitk.calculate_origin_offset(new_spacing, image.header.get_zooms())

    return new_img_like(image, new_data, affine=new_affine)


def normalize(data, opt):
    """follow this https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py"""
    # print("Data min: ", data.min())
    # print("Data max: ", data.max())

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

def iter_volumes(source_dir, transpose_img=False):
    patients_list = glob.glob(os.path.join(source_dir, "*"))

    def iterate_images():
        for idx, patient_dir in enumerate(patients_list):

            patient_name = util_path.get_filename_without_extension(patient_dir)
            patient_name = patient_name.split(".")[0]

            fdata = nib.load(patient_dir).get_fdata()
            depth = fdata.shape[-1]  # Save the number of Slices/Channel

            for d in range(depth):
                img = fdata[:, :, d]  # based on depth index (d): 0000->0128
                if transpose_img:  # fix orientation
                    img = np.transpose(img, [1, 0])
                yield dict(img=img, name=f"{patient_name:s}_{d:05d}", folder_name=f"{patient_name:s}", depth_index=d,   total_depth=depth)

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

def write_to_zip(
        source:                     str,
        dest:                      str,
        dataset:                    str = "RC",
        max_patients:               int = 100000,
):

    def add_to_zip(zipObj, patient):
        files = glob.glob(os.path.join(patient, "*.pickle"))
        if len(files) == 0:
            files = glob.glob(os.path.join(patient, "*.png"))

        print(f">> Writing {patient} to zip file")
        for file in files:
            filename = os.path.join(
                util_path.get_filename_without_extension(patient),
                util_path.get_filename(file),
            )
            # Add file to zip
            zipObj.write(file, filename)

    # Get all patients in temp folder.
    patients = glob.glob(os.path.join(source, "*[!json]"))

    # Get only the names of patients.
    patients = [util_path.get_filename_without_extension(patient) for patient in patients]
    assert len(patients) > 0

    # Create a basename depending on <dataset>, <num_patient>
    max_patients = min(max_patients, len(patients))
    basename = f"{dataset}-num-{max_patients:d}"

    # Shuffle and take max_patients samples from dataset.
    patients = sorted(patients)
    random.Random(max_patients).shuffle(patients)  # comment this line if you don't want to change the order fo the patients across the experiments
    sample_patients = patients[:max_patients]

    # Init zip file.
    out_path = os.path.join(dest, f"{basename}.zip",)

    # Write to zip
    with zipfile.ZipFile(out_path, "w") as zipObj:
        for p in sample_patients:
            patient_path = os.path.join(source, p)
            add_to_zip(zipObj, patient_path)

def open_dataset_patient(source, transpose_img):
    if os.path.isdir(source):
        return iter_volumes(source, transpose_img)
    elif os.path.isfile(source):
        assert False, "unknown archive type"
    else:
        error(f"Missing input file or directory: {source}")

def to_pickle(
        source: str,
        dest: str,
        transpose_img: bool = True,
        is_overwrite: bool = False,
        is_visualize: bool =False
):
    def visualize(x):
        plt.imshow(x, cmap="gray", vmin=x.min(), vmax=x.max())
        plt.show()

    def write_pickle(data, path):
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Open normalized folder.
    num_files, input_iter = open_dataset_patient(source, transpose_img) # Core function.

    # Create a temp folder to be save into zipfile
    temp = os.path.join(dest, "temp")

    if os.path.isdir(temp) and is_overwrite:
        print(f"Removing {temp}")
        shutil.rmtree(temp)
    util_path.create_dir(temp)

    for idx, image in enumerate(input_iter):
        folder_name = image["folder_name"] # patient name
        idx_name = image["name"]
        archive_fname = f"{folder_name}/{idx_name}.pickle"
        util_path.create_dir(os.path.join(temp, f"{folder_name}"))
        out_path = os.path.join(temp, archive_fname)

        img = image["img"]

        # Transform may drop images.
        if img is None:
            break

        # Sanity check
        if random.uniform(0, 1) > 0.9:
            if is_visualize:
                visualize(img)

        # Save the dict as a pickle.
        write_pickle(img, out_path)