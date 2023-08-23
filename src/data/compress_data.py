import shutil

import argparse
argparser = argparse.ArgumentParser(description='Prepare data for compression')
argparser.add_argument('-d', '--dataset_name',
                       help='Dataset name', default='RC')
args = argparser.parse_args()

if __name__ == '__main__':

    # PARAMETERS
    base_name = f'./data/zipped/{args.dataset_name}'
    format = 'zip'
    root_dir = f'./data/processed/{args.dataset_name}/data'



    shutil.make_archive(base_name=base_name,
                        format=format,
                        root_dir=root_dir)