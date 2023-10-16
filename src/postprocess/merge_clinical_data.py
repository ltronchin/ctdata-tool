import numpy as np
import pandas as pd
import os
from datetime import datetime

def load_and_filter_clinical_data(path, patient_dir):


    IDs_processed_volumes = [id for id in os.listdir(patient_dir) if not id.endswith('.xlsx')]
    # clinical data
    pandas_clinical_data = pd.read_csv(os.path.join(path, 'clinical_data.csv'))
    # IDs from the clinical data

    patient_dir = os.path.join(*patient_dir.split('/')[3:])


    if 'AERTS' in path:

        IDs_clinical = pandas_clinical_data['PatientID'].to_list()
        mask_presence = [True if ID_c in IDs_processed_volumes else False for ID_c in IDs_clinical]
        selected_clinical_data = pandas_clinical_data[mask_presence]
        selected_c_and_cols = selected_clinical_data.loc[:,['PatientID', 'Survival.time', 'deadstatus.event']].rename({'Survival.time':'time-day', 'deadstatus.event':'event',  'PatientID': 'ID'}, axis=1)
        selected_c_and_cols['D_class'] = ['AERTS' for i in range(len(selected_c_and_cols))]
        selected_c_and_cols = selected_c_and_cols.astype({'time-day': 'int', 'event': 'int'})
        selected_c_and_cols['directory_data'] = [os.path.join(patient_dir, str(ID)) for ID in selected_c_and_cols['ID']]

    elif 'Radio' in path:
        IDs_clinical = pandas_clinical_data['Case ID'].to_list()
        mask_presence = [True if ID_c in IDs_processed_volumes else False for ID_c in IDs_clinical]
        selected_clinical_data = pandas_clinical_data[mask_presence]
        # Last known alive dates
        dates_alives = selected_clinical_data['Date of Last Known Alive']
        dates_CTs = selected_clinical_data['CT Date']
        date_of_death = selected_clinical_data['Date of Death']

        IDs_clinical_selected = selected_clinical_data['Case ID']

        results = {'ID': [], 'time-day': [], 'event': [], 'D_class': []}
        for alive_d, ct_d, death_d, ID in zip(dates_alives, dates_CTs, date_of_death, IDs_clinical_selected):

            alive_d = datetime.strptime(alive_d, '%m/%d/%Y')
            ct_d = datetime.strptime(ct_d, '%m/%d/%Y')


            if death_d != death_d:
                # Calculate days from CT
                survival_days = (alive_d - ct_d).days
                status = 0

                results['ID'].append(ID)
                results['time-day'].append(survival_days)
                results['event'].append(status)
            else:
                # Calculate days till death
                death_d = datetime.strptime(death_d, '%m/%d/%Y')
                survival_days = (death_d - ct_d).days
                status = 1

                results['ID'].append(ID)
                results['time-day'].append(survival_days)
                results['event'].append(status)
            results['D_class'].append(os.path.basename(path))

        selected_c_and_cols = pd.DataFrame(results)
        selected_c_and_cols = selected_c_and_cols.astype({'time-day': 'int', 'event': 'int'})
        selected_c_and_cols['directory_data'] = [os.path.join(patient_dir, str(ID)) for ID in selected_c_and_cols['ID']]
    elif 'Claro_R' in path:

        IDs_clinical = pandas_clinical_data['CRA'].to_list()
        mask_presence = [True if str(ID_c) in IDs_processed_volumes else False for ID_c in IDs_clinical]
        selected_clinical_data = pandas_clinical_data[mask_presence]

        # Dates:
        Diagnostic_dates = selected_clinical_data['Data Diagnosi']

        ultimo_fup = selected_clinical_data['Ultimo FUP']

        cens_OS = selected_clinical_data['Cens os']

        IDs_clinical_selected = selected_clinical_data['CRA']

        results = {'ID': [], 'time-day': [], 'event': [], 'D_class': []}
        for Diag_d, last_fup, cens_d, ID in zip(Diagnostic_dates, ultimo_fup, cens_OS, IDs_clinical_selected):
            Diag_d = datetime.strptime(Diag_d, '%Y-%m-%d')


            if last_fup != last_fup:
                continue
            else:
                last_fup = datetime.strptime(last_fup, '%Y-%m-%d')
                survival_days = (last_fup - Diag_d).days
                status = cens_d



            results['ID'].append(ID)
            results['time-day'].append(survival_days)
            results['event'].append(status)
            results['D_class'].append('Claro_Retro')

        selected_c_and_cols = pd.DataFrame(results)
        selected_c_and_cols['directory_data'] = [os.path.join(patient_dir, str(ID)) for ID in selected_c_and_cols['ID']]
        selected_c_and_cols = selected_c_and_cols.astype({'time-day': 'int', 'event': 'int'})
    elif 'Claro_P' in path:
        IDs_clinical = pandas_clinical_data['ID paziente'].to_list()
        mask_presence = [True if str(ID_c) in IDs_processed_volumes else False for ID_c in IDs_clinical]
        selected_clinical_data = pandas_clinical_data[mask_presence]


        # Dates:
        Diagnostic_dates = selected_clinical_data['Data Diagnosi']

        ultimo_fup = selected_clinical_data['Ultimo FUP']

        cens_OS = selected_clinical_data['cens OS'].map({'vivo':0, 'morto':1})

        IDs_clinical_selected = selected_clinical_data['ID paziente']

        results = {'ID': [], 'time-day': [], 'event': [], 'D_class': []}
        for Diag_d, last_fup, cens_d, ID in zip(Diagnostic_dates, ultimo_fup, cens_OS, IDs_clinical_selected):
            Diag_d = datetime.strptime(Diag_d, '%Y-%m-%d %H:%M:%S')


            if last_fup != last_fup:
                continue
            else:
                last_fup = datetime.strptime(last_fup, '%Y-%m-%d %H:%M:%S')
                survival_days = (last_fup - Diag_d).days
                status = int(cens_d)



            results['ID'].append(ID)
            results['time-day'].append(survival_days)
            results['event'].append(status)
            results['D_class'].append('Claro_Pro')

        selected_c_and_cols = pd.DataFrame(results)
        selected_c_and_cols['directory_data'] = [os.path.join(patient_dir, str(ID)) for ID in selected_c_and_cols['ID']]
        selected_c_and_cols = selected_c_and_cols.astype({'time-day': 'int', 'event': 'int'})

    return selected_c_and_cols


def merge_clinical(datasets='all'):
    # DATASETS
    server_data = '/Volumes/T7/data'
    interim_directory = os.path.join(server_data, 'interim')
    processed_volume_dir = os.path.join(server_data, 'processed')
    type_of_interpolation = {0: 'volumes_I', 1: 'volumes_V'}
    datasets = [path for path in os.listdir(interim_directory) if 'RC' not in path]
    pandas_clinical_sets = [load_and_filter_clinical_data(os.path.join(interim_directory, dataset),
                            os.path.join(processed_volume_dir, dataset, type_of_interpolation[0])) for dataset in datasets]
    # BOX DATA:
    processed_box_volumes = [pd.read_excel(os.path.join(processed_volume_dir, dataset, type_of_interpolation[0],'data.xlsx' ))
                             for dataset in
                             datasets]
    processed_box_volumes[0].drop(['Unnamed: 0.2','Unnamed: 0.1','Unnamed: 0'], axis=1, inplace=True)
    processed_box_volumes[1].drop(['Unnamed: 0'], axis=1, inplace=True)

    box_Data = pd.concat(processed_box_volumes, axis=0).reset_index(drop=True).rename({'ID': 'PatientID'}, axis=1)

    # Merge clinical data
    pandas_clinical_data = pd.concat(pandas_clinical_sets, axis=0).reset_index(drop=True).rename({'ID': 'PatientID', 'event': 'death_event'}, axis=1)

    # MERGE BOX DATA AND CLINICAL DATA
    pandas_clinical_data = pandas_clinical_data.merge(box_Data, on='PatientID', how='left')

    DeepClusters_IDs = [f'DC_{i}' for i,row in enumerate(pandas_clinical_data.iterrows())]
    pandas_clinical_data['DC_ID'] = DeepClusters_IDs
    pandas_clinical_data.set_index('DC_ID', inplace=True, drop=True)
    pandas_clinical_data.to_csv(os.path.join(server_data, 'processed', 'labels_data.csv'))








if __name__ == '__main__':
    merge_clinical()

