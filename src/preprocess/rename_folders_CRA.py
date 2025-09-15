import os
import pandas as pd
import argparse


argparser = argparse.ArgumentParser(description='Prepare data for training')
argparser.add_argument('-p', '--path', help='path to the folder containing the data', default='./data')
args = argparser.parse_args()




if __name__ == '__main__':

    # Rename folder in the direcotry
    path = '/Volumes/T7/CLARO DATA/Claro_Pro/III_stg/RT_CT/Claro_Pro'
    data_info = '/Volumes/T7/CLARO DATA/Claro_Pro/III_stg/CLARO_PRO_III_CLINIC_23_08_31.xlsx'

    # Get the list of patients
    patients_list = [(os.path.join(path, patient), patient) for patient in os.listdir(path) if os.path.isdir(os.path.join(path, patient))]
    info_data = pd.read_excel(data_info)

    col = info_data.loc[0, :].to_list()
    col[0] = (info_data.loc[1, :].to_list())[0]
    info_data = info_data.drop(index=[0, 1]).reset_index(drop=True)
    info_data.columns = col


    # COLS
    a = '/Volumes/T7/CLARO DATA/Claro_Pro/III_stg/PRO_III_RTCT_Raccolta2023.xlsx'
    ID_checked = pd.read_excel(a, sheet_name='Sheet2')['ID_CHECK'].astype(str).tolist()
    # COLS ID-CRA
    CRA_ID = info_data[['ID paziente', 'CRA']].set_index('CRA').to_dict()['ID paziente']
    IDs = info_data['ID paziente'].to_list()
    CRAs = info_data['CRA'].to_list()

    for path_CRA, CRA in patients_list:
        try:

            #os.rename(path_CRA, os.path.join(path, CRA_ID[CRA]))
            if CRA in ID_checked:
                print(f'Patient {CRA} in ID list!')
            elif CRA in CRAs:
                print(f'Patient {CRA} in CRA list!')
            else:
                print('Patient NOT FOUND {}'.format(CRA))
        except AssertionError as e:
            print(e)
            print(f'Patient NOT FOUND {CRA}')

    patient_ID = [id_ for path, id_ in patients_list]
    correct = [id_ for id_ in ID_checked if id_ not in patient_ID]

















