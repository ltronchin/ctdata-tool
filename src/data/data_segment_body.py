import nibabel as nib
from src.utils import util_contour
from src.utils.util_contour import create_mask_with_largest_contours
from src.utils.util_segmentation import *
from keras.utils import CustomObjectScope
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


# SOME CONFIG TENSORFLOW
print(tf.config.list_physical_devices())
print('tf version:', tf.__version__)
print('available accelerator:', tf.test.gpu_device_name())

# DATASETS
server_data = '/Volumes/T7/data'
datasets = ['NSCLC-RadioGenomics', 'AERTS', 'Claro_Retro', 'Claro_Pro']
type_of_interpolation = {0:'volumes_I', 1: 'volumes_V'}


for dataset in datasets:
    dataset_folder_processed = os.path.join(server_data,  'processed', dataset)
    dataset_I_processed = os.path.join(dataset_folder_processed, type_of_interpolation[0])

    # Load data information
    data_info = pd.read_excel(os.path.join(dataset_I_processed, 'data.xlsx')).drop(['slices_in', 'slices_fin'], axis=1)
    print(data_info.head())


    # Select volumes with missing lungs segmentation
    for ID in data_info['ID']:

        patient_directory = os.path.join(dataset_I_processed, ID)

        # Slices Volume Files
        v_file = os.path.join(patient_directory, 'volume.nii.gz')
        # LOAD VOLUMES
        nii_volume_ = nib.load(v_file)
        nii_volume = nii_volume_.dataobj
        volume_array = np.array(nii_volume)

        mask_body = np.zeros(volume_array.shape).astype(np.int8)
        # Image:


        for z_i in range(volume_array.shape[2]):
            image = volume_array[:, :, z_i]
            # Emulate Batch Data

            body_mask = create_mask_with_largest_contours((image > -300).astype(np.uint8) *255, number_of_contours=1)

            mask_body[:, :, z_i] = body_mask


        # Save bounding box report INTERPOLATED BODY
        bbox_masks_int = {}
        # Save mask

        bbox_mask = util_contour.get_bounding_boxes(volume=mask_body)

        bbox_interpolated_df = pd.DataFrame(bbox_mask).T.drop(columns=['left', 'right']).rename(columns={'max': f'bbox_body'})

        max_bbox = util_contour.get_maximum_bbox_over_slices([value for value in bbox_interpolated_df.loc[:, f'bbox_body'].to_list() if not sum(value) == 0])

        bbox_interpolated_df.loc[:, f'max_bbox_body'] = [max_bbox for i in range(len(bbox_interpolated_df))]

        bbox_masks_int['Body'] = bbox_interpolated_df

        pass

        data_bbox = pd.read_excel(os.path.join(dataset_I_processed, 'data.xlsx')).drop('Unnamed: 0', axis=1)


