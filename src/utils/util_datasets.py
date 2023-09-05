import glob
import operator
import os
import re
import pydicom as dicom
import numpy as np
import pandas as pd
import pydicom

from src.utils import util_contour, util_dicom


class BaseDataset(object):

    def __init__(self, cfg):
        self.shape = None
        self.voxel_by_rois = None
        self.mask_dir = None
        self.rois_name_dict = {}
        self.structures = {}
        self.rois_classes = []
        self.volumes_target = None
        self.dicom_info = None
        self.patient_paths = None
        self.patient_ids = None
        self.labels = None
        self.metadata = None
        self.dataset_name = cfg['data']['dataset_name']
        self.label_file = cfg['data']['label_file']
        self.img_raw_dir = os.path.join(cfg['data']['img_dir'], self.dataset_name)
        self.interim_dir = os.path.join(cfg['data']['interim_dir'], self.dataset_name)
        self.processed_dir = os.path.join(cfg['data']['processed_dir'], self.dataset_name)
        self.reports_dir = os.path.join(cfg['reports']['reports_dir'], self.dataset_name)
        self.masks_target = cfg['data']['contour']['masks_target']
        self.union_target = cfg['data']['contour']['union_target']
        self.dicom_tags = cfg['data']['dicom_tags']
        self.metadata_file = cfg['data']['metadata_file'] if 'None' not in cfg['data']['metadata_file'] else None
        self.ordered_slices = None
        # Function running at __init__

        # Load label file
        self.load_label()
        # Load Metadata file
        self.load_metadata()

    # Methods Getter/Setter

    def create_check_directories(self):
        for dir in [self.interim_dir, self.processed_dir, self.reports_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def load_label(self):
        if self.label_file is None:
            self.labels = None
        else:
            self.labels = pd.read_excel(self.label_file) if self.label_file.endswith('.xlsx') else pd.read_csv(self.label_file, sep=';')

    def get_label(self):
        return self.labels

    def load_metadata(self):
        if self.metadata_file is None:
            self.metadata = None
        else:
            self.metadata = pd.read_excel(self.metadata_file) if self.metadata_file.endswith('.xlsx') else pd.read_csv(self.metadata_file, sep=',')

    def get_metadata(self):
        return self.metadata

    def get_patients_directories(self):
        raise NotImplementedError(f"The method get_patients_directories is not implemented for the child class: {self.__class__.__name__}")

    def create_dicom_info_report(self):
        self.dicom_info = pd.DataFrame(columns=self.dicom_tags + ['#slices'])
        return self

    def save_dicom_info_report(self):
        self.dicom_info.to_excel(os.path.join(self.interim_dir, 'patients_info.xlsx'), index=False)

    def create_save_structures_report(self, data_structures):
        df = pd.DataFrame(data_structures)
        df.to_excel(os.path.join(self.interim_dir, 'structures.xlsx'), index=False)

    def load_structures_report(self):
        self.structures = pd.read_excel(os.path.join(self.interim_dir, 'structures.xlsx')).set_index('patient_dir')
        return self

    def get_structures(self):
        return self.structures

    def save_clinical(self):
        raise NotImplementedError(f"The method save_clinical is not implemented for the child class: {self.__class__.__name__}")

    def load_dicom_info_report(self):
        self.dicom_info = pd.read_excel(os.path.join(self.interim_dir, 'patients_info.xlsx')).set_index('PatientID')
        return self

    def get_dicom_info(self):
        return self.dicom_info

    def add_dicom_infos(self, dicom_files, patient_id):
        dicom_files.sort()
        # Read the first DICOM file in the directory and extract the DICOM tags
        ds = pydicom.dcmread(dicom_files[0])
        data = {
            tag: getattr(ds, tag, None) for tag in self.dicom_tags
        }
        data['#slices'] = len(dicom_files)
        assert data['SliceThickness'] >= 1, f'Patient {patient_id} has SliceThickness < 1'

        self.dicom_info.loc[len(self.dicom_info)] = data

        return data

    def get_dicom_files(self, patient_dir, segmentation_load=False):
        raise NotImplementedError(f"The method get_dicom_files is not implemented for the child class: {self.__class__.__name__}")

    def get_structures_names(self, ds_seg):
        # Initialize structures ROIS names
        self.initialize_rois()
        # Available structures
        for item in ds_seg.StructureSetROISequence:
            name = item.ROIName
            matching, roi_class = self.matching_rois(roi_name=name)
            assert matching is not None
            if matching:
                self.structures[item.ROINumber] = name
                self.rois_classes.append(roi_class)
        if len(self.structures) == 0:
            print("No structures found")
        else:
            print("Available structures: ", self.structures)
        return self

    def get_structures_and_classes(self):
        return self.structures, self.rois_classes

    def get_rois_name_dict(self):
        pattern_lung = [re.compile('(^lung[-_\s])', re.IGNORECASE), re.compile('(^polmone[\s])', re.IGNORECASE), re.compile('(^lungs[-_\s])', re.IGNORECASE)]
        pattern_body = [re.compile('(^body[\s])', re.IGNORECASE), re.compile('(^corpo[\s])', re.IGNORECASE),
                        re.compile('(^external[\s])', re.IGNORECASE)]
        pattern_ctv = [re.compile('^ctv[0-9]{0,1}', re.IGNORECASE)]
        pattern_gtv = [re.compile('(^gtv[0-9-_\s])', re.IGNORECASE)]

        self.rois_name_dict = {'lung': pattern_lung, 'body': pattern_body, 'ctv': pattern_ctv, 'gtv': pattern_gtv}

    def matching_rois(self, roi_name=None):
        return None, None

    def set_target_masks(self, rois_dict, masks_target, union_target, shape):
        pass

    def initialize_rois(self):
        self.get_rois_name_dict()

    def initialize_contour_analysis(self):
        self.volumes_target = {}
        self.mask_dir = os.path.join(self.interim_dir, '3D_masks')

    def get_mask_dir(self):
        return self.mask_dir

    def create_masks_dictionary(self, rois_dict, shape):
        """
        This function creates the dictionary of the target volumes masks

        :param shape:
        :return:
        """

        # Create empty dictionary for the target volumes masks
        bool_Lungs = False
        bool_Lesions = False
        volumes_target = {name_target: np.zeros(
            shape=shape) for name_target in self.masks_target}
        for roi_name, roi_mask in rois_dict.items():
            matching, name_volume = self.matching_rois(roi_name)
            if matching and name_volume == 'Lungs':
                bool_Lungs = True
            elif matching and name_volume == 'Lesions':
                bool_Lesions = True
            if matching and name_volume in self.masks_target:
                # Stack the mask slices in the third dimension and convert to boolean
                roi_mask = self.roi_volume_stacking(roi_mask)
                # get volumes target and convert to boolean
                volumes_target[name_volume] = volumes_target[name_volume] > 0.5
                # Create the union of all the rois by the common target volumes
                roi_masks_union = util_contour.Volume_mask_and_or_mask(volumes_target[name_volume], roi_mask, OR=True)
                # Update the dictionary
                volumes_target[name_volume] = roi_masks_union

        if not bool_Lungs:
            self.masks_target = [mask for mask in self.masks_target if mask != 'Lungs']
            volumes_target = {name_target: volume for name_target, volume in volumes_target.items() if name_target != 'Lungs'}
        # Union Target
        if len(self.union_target) > 0 and bool_Lesions and bool_Lungs:
            names_volumes = self.union_target[0].split('_')
            # Create the union of all the rois by the common target volumes
            volumes_target[self.union_target[0]] = util_contour.Volume_mask_and_or_mask(volumes_target[names_volumes[0]] > 0.5,
                                                                                        volumes_target[names_volumes[1]] > 0.5, OR=True)
            self.masks_target = self.masks_target + self.union_target

        # Convert to np.uint8
        self.volumes_target = {name_target: volume.astype(np.int8) * 255 for name_target, volume in volumes_target.items()}

        return self.volumes_target, self.masks_target

    def roi_volume_stacking(self, roi_mask):
        return np.stack(roi_mask, axis=2) > 0.5

    def get_slices_dict(self, slices_dir):
        if slices_dir[-1] != '/': slices_dir += '/'
        slices = []
        for s in os.listdir(slices_dir):
            try:
                f = dicom.read_file(slices_dir + '/' + s)
                assert f.Modality != 'RTDOSE'
                slices.append((f, s.split('.')[0]))

            except:
                continue
        slice_dict = {(s.SOPInstanceUID, namefile): s.ImagePositionPatient[-1] for (s, namefile) in slices}
        return slice_dict

    def set_slices_dict(self, ordered_slices):
        self.ordered_slices = ordered_slices

    def get_ordered_slices_dict(self):
        return self.ordered_slices

    def get_slice_file(self, slices_dir, img_id, img_SOP=None):
        CT_dir_and_name = slices_dir + "/"
        return CT_dir_and_name + img_id[1] + ".dcm"

    def set_shape(self, shape):
        self.shape = shape


    def create_voxels_by_rois(self, ds_seg, roi_names, slices_dir_patient, img_id, number_of_slices=None):

        self.voxel_by_rois = {name: [] for name in roi_names}

        for roi_name in roi_names:
            # GET ROIS
            idx = np.where(np.array(util_dicom.get_roi_names(ds_seg)) == roi_name)[0][0]
            contour_datasets = util_dicom.get_roi_contour_ds(ds_seg, idx)
            mask_dict = util_dicom.get_mask_dict(contour_datasets, slices_dir_patient, img_id=img_id, dataset=self)
            if img_id in mask_dict:
                mask_array = mask_dict[img_id]
            else:
                mask_array = np.zeros(shape=self.shape)
            self.voxel_by_rois[roi_name].append(mask_array)
        return self

    def get_coord(self, contour_coord, img_id, img_SOP=None):
        coord = []
        for i in range(0, len(contour_coord), 3):
            coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))
        return coord, img_id

    def get_voxel_by_rois(self):
        return self.voxel_by_rois


class RECO(BaseDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_patients_directories(self):
        patient_list_accepted = os.listdir(self.img_raw_dir)
        self.patient_paths = [os.path.join(self.img_raw_dir, Id) for Id in os.listdir(self.img_raw_dir) if Id in patient_list_accepted]
        self.patient_ids = [os.path.basename(patient_path) for patient_path in self.patient_paths]
        return self.patient_paths, self.patient_ids

    def load_label(self):
        if self.label_file is None:
            self.labels = None
        else:
            self.labels = pd.read_excel(self.label_file) if self.label_file.endswith('.xls') else pd.read_csv(self.label_file, sep=';')
            self.labels.drop(columns=['Unnamed: 0'], inplace=True)

    def get_dicom_files(self, patient_dir, segmentation_load=False):
        # List all .dcm files in the patient directory
        CT_files = glob.glob(os.path.join(patient_dir, 'CT*.dcm'))
        if segmentation_load:
            seg_files = glob.glob(os.path.join(patient_dir, 'RS*.dcm'))
            assert len(seg_files) > 0, "No segmentation file found"
            return CT_files, patient_dir, seg_files, patient_dir
        else:
            return CT_files, patient_dir, None, None

    def add_dicom_infos(self, dicom_files, patient_id):

        dicom_temp = self.dicom_info.copy()
        data = super().add_dicom_infos(dicom_files, patient_id)
        # Get label
        label = self.labels.loc[self.labels['Nome paziente'] == patient_id, 'Risposta Completa (0 no, 1 si)'].values[0]

        data['RC'] = label
        dicom_temp.loc[len(dicom_temp)] = data
        self.dicom_info = dicom_temp.copy()
        return data

    def matching_rois(self, roi_name=None):
        roi_name = roi_name.lower()
        if any(pattern.search(roi_name) for pattern in self.rois_name_dict['lung']):
            return True, 'Lungs'
        elif any(pattern.search(roi_name) for pattern in self.rois_name_dict['body']):
            return True, 'Body'
        elif any(pattern.search(roi_name) for pattern in self.rois_name_dict['ctv']):
            return True, 'Lesions'
        else:
            return False, None

    def get_slices_dict(self, slices_dir):
        if slices_dir[-1] != '/': slices_dir += '/'
        slices = []
        for s in os.listdir(slices_dir):
            try:
                f = dicom.read_file(slices_dir + '/' + s)
                f.ImagePositionPatient  #
                assert f.Modality != 'RTDOSE'
                slices.append(f)
            except:
                continue
        slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}

        return slice_dict

    def get_slice_file(self, slices_dir, img_id, img_SOP=None):
        img_SOP = img_id
        CT_dir_and_name = slices_dir + "/CT."
        return CT_dir_and_name + img_id + ".dcm"


class AERTS(BaseDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_patients_directories(self):
        patient_list_accepted = self.labels['PatientID'].to_list()
        self.patient_paths = [os.path.join(self.img_raw_dir, Id) for Id in os.listdir(self.img_raw_dir) if Id in patient_list_accepted]
        self.patient_ids = [os.path.basename(patient_path) for patient_path in self.patient_paths]
        return self.patient_paths, self.patient_ids

    def matching_rois(self, roi_name=None):
        if any(pattern.search(roi_name.lower()) for pattern in self.rois_name_dict['lung']):
            return True, 'Lungs'
        elif any(pattern.search(roi_name.lower()) for pattern in self.rois_name_dict['gtv']):
            return True, 'Lesions'
        else:
            return False, None

    def get_dicom_files(self, patient_dir, segmentation_load=False):
        # Patient metadata
        patient_id = os.path.basename(patient_dir)
        patient_metadata = self.metadata.loc[self.metadata['Data Description URI'] == patient_id]
        # CT scan directory
        CT_dir = patient_metadata[patient_metadata['Manufacturer'] == 'CT']['File Location'].values[
                     0].split('\\')[-2:]
        CT_dir = os.path.join(patient_dir, *CT_dir)

        # find all the .dicom files in the CT scan directory
        CT_files = glob.glob(os.path.join(CT_dir, '*.dcm'))

        if segmentation_load:
            # Segmentation file
            SEG_dir = patient_metadata[patient_metadata['Manufacturer'] == 'RTSTRUCT']['File Location'].values[
                          0].split('\\')[-2:]
            SEG_dir = os.path.join(patient_dir, *SEG_dir)
            # Open files
            seg_files = glob.glob(os.path.join(SEG_dir, '*.dcm'))
            assert len(seg_files) > 0, "No segmentation file found"
            return CT_files, CT_dir, seg_files, SEG_dir

        else:
            # No segmentation files required
            return CT_files, CT_dir, None, None





class NSCLCRadioGenomics(BaseDataset):

    def __init__(self, cfg):
        super().__init__(cfg)

    def get_patients_directories(self):
        patient_list_accepted = self.labels['Case ID'].to_list()
        self.patient_paths = [os.path.join(self.img_raw_dir, Id) for Id in os.listdir(self.img_raw_dir) if Id in patient_list_accepted]
        self.patient_ids = [os.path.basename(patient_path) for patient_path in self.patient_paths]
        return self.patient_paths, self.patient_ids

    def get_dicom_files(self, patient_dir, segmentation_load=False):

        patient_id = os.path.basename(patient_dir)
        # Patient metadata selection
        patient_metadata = self.get_metadata().loc[self.get_metadata()['Data Description URI'] == patient_id]

        # CT scan directory

        all_CT_scans = patient_metadata[patient_metadata['Manufacturer'] == 'CT']
        selected_cts = all_CT_scans[['coronals' not in str(CT_study_date).lower() and SOP_class > 1 for CT_study_date, SOP_class in
                                     zip(all_CT_scans.loc[:, 'Study Date'].to_list(), all_CT_scans.loc[:, 'SOP Class UID'].to_list())]]

        if len(selected_cts) > 1:
            select_chest = selected_cts[['thorax' in str(CT_study_date).lower() or 'chest' in str(CT_study_date).lower() or 'lung' in str(CT_study_date).lower()
                                         for CT_study_date in (selected_cts.loc)[:, 'Study Date'].to_list()]]

            assert len(select_chest) > 0, "No chest CT found"
            CT_thorax_scan_dir = select_chest['File Location'].values[0].split('\\')[-2:]
        else:
            CT_thorax_scan_dir = selected_cts['File Location'].values[0].split('\\')[-2:]

        # Final scan directory
        CT_dir = os.path.join(patient_dir, *CT_thorax_scan_dir)

        # find all the .dicom files in the CT scan directory
        CT_files = glob.glob(os.path.join(CT_dir, '*.dcm'))

        if segmentation_load:
            # Segmentation file
            assert 'SEG' in patient_metadata['Manufacturer'].to_list(), "No segmentation file found, SEG not in {}".format(patient_metadata['Manufacturer'].to_list())
            SEG_dir = patient_metadata[patient_metadata['Manufacturer'] == 'SEG']['File Location'].values[
                          0].split('\\')[-2:]
            SEG_dir = os.path.join(patient_dir, *SEG_dir)
            # Open files
            seg_files = glob.glob(os.path.join(SEG_dir, '*.dcm'))
            assert len(seg_files) > 0, "No segmentation file found"
            return CT_files, CT_dir, seg_files, SEG_dir
        else:
            # No segmentation file
            return CT_files, CT_dir, None, None

    def get_structures_names(self, ds_seg):
        self.initialize_rois()
        # We need to handle this dataset in a different way
        matching, roi_class = self.matching_rois()
        if matching:
            self.structures['0'] = 'NSCLC-Lung'
            self.rois_classes.append(roi_class)
        return self

    def matching_rois(self, roi_name=None):
        return True, 'Lesions'

    def roi_volume_stacking(self, roi_mask):
        return roi_mask[0] > 0.5

    def create_voxels_by_rois(self, ds_seg, roi_names, slices_dir_patient, img_id, number_of_slices=None):

        slice_order = self.get_ordered_slices_dict()
        for roi_name in roi_names:
            seg = ds_seg.pixel_array
            # reorient the seg array
            seg = np.fliplr(seg.T)

            if seg.shape[2] != number_of_slices:
                # get the ReferencedSOPInstanceUID from the RTSTRUCT file
                s = ds_seg.ReferencedSeriesSequence._list
                RefSOPInstanceUID_list = [slice.ReferencedSOPInstanceUID for slice in s[0].ReferencedInstanceSequence._list]
                true_slices = [img_id[0] in RefSOPInstanceUID_list for img_id, UID in zip(slice_order, RefSOPInstanceUID_list)]
                new_voxel = {}
                i = 0

                for img_id in slice_order:
                    found_ = False
                    for UID in RefSOPInstanceUID_list:
                        if str(img_id[0][0]) == (UID):
                            new_voxel[img_id[0][1]] = seg[:, :, i]
                            i = + 1
                            found_ = True
                        else:
                            pass
                    if not found_:
                        new_voxel['None' + img_id[0][1]] = np.zeros(shape=(512, 512))

                seg = np.stack(list(new_voxel.values()), 2)

            self.voxel_by_rois[roi_name].append(seg)
        return self

    def get_coord(self, contour_coord, img_id, img_SOP=None):
        if str(img_id[0]) != str(img_SOP):
            coord = []
            for i in range(0, len(contour_coord), 3):
                coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))
        else:
            coord = None
        return coord, img_id

def create_masks_dictionary(rois_dict, masks_target, union_target, dataset_name, shape):
    """
    This function creates the dictionary of the target volumes masks
    :param rois_dict:
    :param masks_target:
    :param union_target:
    :param dataset_name:
    :param shape:
    :return:
    """

    # Create empty dictionary for the target volumes masks
    bool_Lungs = False
    volumes_target = {name_target: np.zeros(
        shape=shape) for name_target in masks_target}
    for roi_name, roi_mask in rois_dict.items():
        matching, name_volume = rois_matching(dataset_name, roi_name)
        if matching and name_volume == 'Lungs':
            bool_Lungs = True
        elif matching and name_volume == 'Lesions':
            bool_Lesions = True
        if matching and name_volume in masks_target:
            # Stack the mask slices in the third dimension and convert to boolean
            roi_mask = np.stack(roi_mask, axis=2) > 0.5 if dataset_name != 'NSCLC-RadioGenomics' else roi_mask[0] > 0.5
            # get volumes target and convert to boolean
            volumes_target[name_volume] = volumes_target[name_volume] > 0.5
            # Create the union of all the rois by the common target volumes
            roi_masks_union = util_contour.Volume_mask_and_or_mask(volumes_target[name_volume], roi_mask, OR=True)
            # Update the dictionary
            volumes_target[name_volume] = roi_masks_union

    if not bool_Lungs:
        masks_target = [mask for mask in masks_target if mask != 'Lungs']
        volumes_target = {name_target: volume for name_target, volume in volumes_target.items() if name_target != 'Lungs'}
    # Union Target
    if len(union_target) > 0 and bool_Lesions and bool_Lungs:
        names_volumes = union_target[0].split('_')
        # Create the union of all the rois by the common target volumes
        volumes_target[union_target[0]] = util_contour.Volume_mask_and_or_mask(volumes_target[names_volumes[0]] > 0.5,
                                                                               volumes_target[names_volumes[1]] > 0.5, OR=True)
        masks_target = masks_target + union_target

    # Convert to np.uint8
    volumes_target = {name_target: volume.astype(np.int8) * 255 for name_target, volume in volumes_target.items()}

    return volumes_target, masks_target


"""def get_structures_names(ds_seg, dataset_name):
    # https://github.com/pydicom/pydicom/issues/961

    # Available structures
    structures = {}
    rois_classes = []
    # We need to handle this dataset in a different way
    if dataset_name == 'NSCLC-RadioGenomics':
        matching, roi_class = rois_matching(dataset_name)
        if matching:
            structures['0'] = 'NSCLC-Lung'
            rois_classes.append(roi_class)
    else:
        for item in ds_seg.StructureSetROISequence:
            name = item.ROIName
            matching, roi_class = rois_matching(dataset_name, name)
            if matching:
                structures[item.ROINumber] = name
                rois_classes.append(roi_class)

    if len(structures) == 0:
        print("No structures found")
    else:
        print("Available structures: ", structures)
    return structures, rois_classes"""

"""def rois_matching(dataset_name, roi_name=None):
    pattern_lung = [re.compile('(^lung[-_\s])', re.IGNORECASE), re.compile('(^polmone[\s])', re.IGNORECASE), re.compile('(^lungs[-_\s])', re.IGNORECASE)]
    pattern_body = [re.compile('(^body[\s])', re.IGNORECASE), re.compile('(^corpo[\s])', re.IGNORECASE),
                    re.compile('(^external[\s])', re.IGNORECASE)]
    pattern_ctv = [re.compile('^ctv[0-9]{0,1}', re.IGNORECASE)]
    pattern_gtv = [re.compile('(^gtv[0-9-_\s])', re.IGNORECASE)]

    if dataset_name == 'AERTS':
        if any(pattern.search(roi_name.lower()) for pattern in pattern_lung):
            return True, 'Lungs'
        elif any(pattern.search(roi_name.lower()) for pattern in pattern_gtv):
            return True, 'Lesions'
        else:
            return False, None
    elif dataset_name == 'RC':

        if any(pattern.search(roi_name) for pattern in pattern_lung):
            return True, 'Lungs'
        elif any(pattern.search(roi_name) for pattern in pattern_body):
            return True, 'Body'
        elif any(pattern.search(roi_name) for pattern in pattern_ctv):
            return True, 'Lesions'
        else:
            return False
    elif dataset_name == 'NSCLC-RadioGenomics':
        # Get structures
        return True, 'Lesions'"""

"""def get_patients_directories(dataset_name, img_dir, info_file, metadata_file, return_patient_ID=True, label_file=None):
    if label_file == None:
        label_file = info_file
    elif info_file == None:
        info_file = label_file
    else:
        raise ValueError("Only one of info_file or label_file can be None")
    if dataset_name == 'AERTS':
        print('AERTS')
        df_labels = pd.read_excel(info_file) if info_file.endswith('.xlsx') else pd.read_csv(info_file, sep=';')
        patientID_key = 'PatientID'
        patient_list_accepted = df_labels['PatientID'].to_list()
        # Metadata for CT scan
        metadata = pd.read_csv(metadata_file, sep=',')
    elif dataset_name == 'NSCLC-RadioGenomics':
        print('NSCLC-RadioGenomics')
        df_labels = pd.read_excel(info_file) if info_file.endswith('.xlsx') else pd.read_csv(info_file, sep=';')
        patientID_key = 'PatientID'
        patient_list_accepted = df_labels[patientID_key].to_list()
        metadata = pd.read_csv(metadata_file, sep=',')
    elif dataset_name == 'RC':
        # List all patient directories
        patient_list_accepted = os.listdir(img_dir)
        metadata = None

    patient_list = [os.path.join(img_dir, dir) if not return_patient_ID else dir for dir in os.listdir(img_dir) if dir in patient_list_accepted]

    return patient_list, metadata"""

"""def get_dicom_files(dataset_name, metadata, patient_dir, segmentation_load=True):
    if dataset_name == 'AERTS':
        # Patient metadata
        patient_id = os.path.basename(patient_dir)
        patient_metadata = metadata.loc[metadata['Data Description URI'] == patient_id]
        # CT scan directory
        CT_scan_dir = patient_metadata[patient_metadata['Manufacturer'] == 'CT']['File Location'].values[
                          0].split('\\')[-2:]
        CT_scan_dir = os.path.join(patient_dir, *CT_scan_dir)

        # find alll the .dicom files in the CT scan directory
        dicom_files = glob.glob(os.path.join(CT_scan_dir, '*.dcm'))
        if segmentation_load:
            # Segmentation file
            RTSTRUCT_dir = patient_metadata[patient_metadata['Manufacturer'] == 'RTSTRUCT']['File Location'].values[
                               0].split('\\')[-2:]
            RTSTRUCT_dir = os.path.join(patient_dir, *RTSTRUCT_dir)
            # Open files
            seg_files = glob.glob(os.path.join(RTSTRUCT_dir, '*.dcm'))
            assert len(seg_files) > 0, "No segmentation file found"
        else:
            seg_files = None
            RTSTRUCT_dir = None
        return dicom_files, CT_scan_dir, seg_files, RTSTRUCT_dir


    elif dataset_name == 'RC':
        # List all .dcm files in the patient directory
        dicom_files = glob.glob(os.path.join(patient_dir, 'CT*.dcm'))
        if segmentation_load:
            seg_files = glob.glob(os.path.join(patient_dir, 'RS*.dcm'))
        else:
            seg_files = None
            RTSTRUCT_dir = None
        return dicom_files, patient_dir, seg_files, patient_dir

    elif dataset_name == 'NSCLC-RadioGenomics':
        patient_id = os.path.basename(patient_dir)
        patient_metadata = metadata.loc[metadata['Data Description URI'] == patient_id]

        # CT scan directory
        if patient_id == 'R01-011':
            pass
        all_CT_scans = patient_metadata[patient_metadata['Manufacturer'] == 'CT']
        selected_cts = all_CT_scans[['coronals' not in str(CT_study_date).lower() and SOP_class > 1 for CT_study_date, SOP_class in
                                     zip(all_CT_scans.loc[:, 'Study Date'].to_list(), all_CT_scans.loc[:, 'SOP Class UID'].to_list())]]

        if len(selected_cts) > 1:
            select_chest = selected_cts[['thorax' in str(CT_study_date).lower() or 'chest' in str(CT_study_date).lower() or 'lung' in str(CT_study_date).lower() for CT_study_date in (
                                                                                                                                                                                          selected_cts.loc)[
                                                                                                                                                                                      :,
                                                                                                                                                                                      'Study Date'].to_list()]]

            assert len(select_chest) > 0, "No chest CT found"

            CT_thorax_scan_dir = select_chest['File Location'].values[0].split('\\')[-2:]
        else:
            CT_thorax_scan_dir = selected_cts['File Location'].values[0].split('\\')[-2:]

        CT_scan_dir = os.path.join(patient_dir, *CT_thorax_scan_dir)

        # find alll the .dicom files in the CT scan directory
        dicom_files = glob.glob(os.path.join(CT_scan_dir, '*.dcm'))

        if segmentation_load:
            # Segmentation file
            if 'SEG' in patient_metadata['Manufacturer'].to_list():
                pass
            assert 'SEG' in patient_metadata['Manufacturer'].to_list(), "No segmentation file found, SEG not in {}".format(patient_metadata['Manufacturer'].to_list())
            RTSTRUCT_dir = patient_metadata[patient_metadata['Manufacturer'] == 'SEG']['File Location'].values[
                               0].split('\\')[-2:]
            RTSTRUCT_dir = os.path.join(patient_dir, *RTSTRUCT_dir)
            # Open files
            seg_files = glob.glob(os.path.join(RTSTRUCT_dir, '*.dcm'))
            assert len(seg_files) > 0, "No segmentation file found"
        else:
            seg_files = None
            RTSTRUCT_dir = None
        return dicom_files, CT_scan_dir, seg_files, RTSTRUCT_dir
"""
