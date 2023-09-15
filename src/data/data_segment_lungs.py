import nibabel as nib
from src.utils import util_contour
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
datasets = ['AERTS', 'NSCLC-RadioGenomics']
type_of_interpolation = {0:'volumes_I', 1: 'volumes_V'}

# Saved Model UNET:
trained_model_path = model_path = "./weights/Lung_Segmentation_2D_13E.h5"  # os.path.join(abs_path, "saved_models", f"TFmodel_lung224_21S.h5")
trained_model = load_saved_model(trained_model_path)


for dataset in datasets:
    dataset_folder_processed = os.path.join(server_data,  'processed', dataset)
    dataset_I_processed = os.path.join(dataset_folder_processed, type_of_interpolation[0])

    # Load data information
    data_info = pd.read_excel(os.path.join(dataset_I_processed, 'data.xlsx'))
    print(data_info.head())
    if 'NSCLC-RadioGenomics' in dataset:

        for name in ['max_bbox_lesions', 'max_bbox_lungs', 'max_bbox_lesions_lungs']:
            data_info[name] = None

    # Select volumes with missing lungs segmentation
    selection = [True if 'Lungs' not in eval(ROIS_for_ID) else False for ROIS_for_ID in data_info['ROIs_names']]
    data_info_missing_lungs = data_info[selection]

    for ID, index in zip(data_info_missing_lungs['ID'], data_info_missing_lungs.index.tolist()):

        patient_directory = os.path.join(dataset_I_processed, ID)

        # Slices Volume Files
        v_file = os.path.join(patient_directory, 'volume.nii.gz')
        le_file = os.path.join(patient_directory, 'lesions_mask.nii.gz')

        lu_file = os.path.join(patient_directory, 'lungs_mask.nii.gz')
        lu_le_file = os.path.join(patient_directory, 'lesions_lungs_mask.nii.gz')

        # LOAD VOLUMES
        nii_volume_ = nib.load(v_file)
        nii_volume = nii_volume_.dataobj
        volume_array = np.array(nii_volume)

        nii_lesion_ = nib.load(le_file)
        nii_lesion = nii_lesion_.dataobj
        lesion_array = np.array(nii_lesion)

        mask_lungs = np.zeros(volume_array.shape).astype(np.int8)
        # Image:

        for z_i in range(volume_array.shape[2]):
            image = Image.fromarray(volume_array[:, :, z_i])
            image = image.resize((224, 224))
            # Emulate Batch Data

            image_data = np.array(image)[np.newaxis, :, :, np.newaxis]
            # INFERENCE
            pred = trained_model.predict(image_data)
            # SELECTION AND RESHAPE
            mask_slice = Image.fromarray(np.array(pred[0, :, :, 0] > 0.5))
            mask_slice = mask_slice.resize((512, 512))
            mask_slice = np.array(mask_slice).astype(np.int8)
            mask_lungs[:, :, z_i] = mask_slice


        mask_lungs = mask_lungs.astype(np.int8) * 255

        # Create Intersection:
        volume_lesion_lungs = util_contour.Volume_mask_and_or_mask(mask_lungs > 0.5, lesion_array > 0.5, OR=True)
        volume_lesion_lungs = volume_lesion_lungs.astype(np.int8) * 255

        # Save bounding box report INTERPOLATED BODY
        bbox_masks_int = {}
        dict_final_mask = {'Lesions': lesion_array, 'Lungs': mask_lungs, 'Lesions_Lungs': volume_lesion_lungs}
        for mask_name, mask  in dict_final_mask.items():
            bbox_mask = util_contour.get_bounding_boxes(volume=mask)

            bbox_interpolated_df = pd.DataFrame(bbox_mask).T.drop(columns=['left', 'right']).rename(columns={'max': f'bbox_{mask_name.lower()}'})

            max_bbox = util_contour.get_maximum_bbox_over_slices([value for value in bbox_interpolated_df.loc[:, f'bbox_{mask_name.lower()}'].to_list() if not sum(value) == 0])

            bbox_interpolated_df.loc[:, f'max_bbox_{mask_name.lower()}'] = [max_bbox for i in range(len(bbox_interpolated_df))]

            bbox_masks_int[mask_name] = bbox_interpolated_df
        # Concatenate bounding box report INTERPOLATED for BODY and LUNGS
        df_interpolated = pd.concat(
            [bbox_df for bbox_df in bbox_masks_int.values()],
            axis=1)

        selection_slices_with_lungs = [True if bbox != [0, 0, 0, 0] else False for bbox in df_interpolated['bbox_lungs'] ]
        dict_max_bbox = {
            f'max_bbox_{mask_class.lower()}': util_contour.get_maximum_bbox_over_slices([bbox for bbox in df_interpolated.loc[selection_slices_with_lungs,
            f'bbox_{mask_class.lower()}'].tolist() if sum(bbox) != 0]) for mask_class in dict_final_mask.keys()

        }
        for key in dict_max_bbox.keys():
            data_info.loc[index, key] = str(dict_max_bbox[key])
        data_info.loc[index, '#slices_lungs_only'] = int(len([0 for i in selection_slices_with_lungs if i]))

        first_slices_lung = [i for i, slice in enumerate(selection_slices_with_lungs) if slice]
        data_info.loc[index, 'slices_in'] = first_slices_lung[0]
        data_info.loc[index, 'slices_fin'] = first_slices_lung[0] + 1
        data_info.loc[index, 'ROIs_names'] = str(list(dict_final_mask.keys()))

        # Save new volumes:
        # LUNGS
        mask_lungs_sel = mask_lungs[:,:,selection_slices_with_lungs]
        mask_lungs_nii = nib.Nifti1Image(mask_lungs_sel, nii_lesion_.affine, nii_lesion_.header)
        nib.save(mask_lungs_nii, lu_file)
        # LESIONS-LUNGS
        volume_lesion_lungs_sel = volume_lesion_lungs[:, :, selection_slices_with_lungs]
        mask_lungs_lesion_nii = nib.Nifti1Image(volume_lesion_lungs_sel, nii_lesion_.affine, nii_lesion_.header)
        nib.save(mask_lungs_lesion_nii, lu_le_file)

        # VOLUME TOT
        volume_sel = volume_array[:,:, selection_slices_with_lungs]
        volume_nii = nib.Nifti1Image(mask_lungs_sel, nii_volume_.affine, nii_volume_.header)
        nib.save(volume_nii, v_file)
        # LESIONS

        lesion_sel = lesion_array[:, :, selection_slices_with_lungs]
        lesion_nii = nib.Nifti1Image(lesion_sel, nii_lesion_.affine, nii_lesion_.header)
        nib.save(lesion_nii, le_file)

    data_info.to_excel(os.path.join(dataset_I_processed, 'data.xlsx'))

