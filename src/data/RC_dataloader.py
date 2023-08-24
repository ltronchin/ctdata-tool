# Torch DataLoader for RC dataset


import os
import cv2
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

def get_box(img, box, perc_border=.0):
    # Sides
    l_h = box[2] - box[0]
    l_w = box[3] - box[1]
    # Border
    diff_1 = math.ceil((abs(l_h - l_w) / 2))
    diff_2 = math.floor((abs(l_h - l_w) / 2))
    border = int(perc_border * diff_1)
    # Img dims
    img_h = img.shape[0]
    img_w = img.shape[1]
    if l_h > l_w:
        if box[0]-border < 0:
            pad = 0-(box[0]-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+border > img_h:
            pad = (box[2]+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-diff_1-border < 0:
            pad = 0-(box[1]-diff_1-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+diff_2+border > img_w:
            pad = (box[3]+diff_2+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-border:box[2]+border, box[1]-diff_1-border:box[3]+diff_2+border]
    elif l_w > l_h:
        if box[0]-diff_1-border < 0:
            pad = 0-(box[0]-diff_1-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+diff_2+border > img_h:
            pad = (box[2]+diff_2+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-border < 0:
            pad = 0-(box[1]-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+border > img_w:
            pad = (box[3]+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-diff_1-border:box[2]+diff_2+border, box[1]-border:box[3]+border]
    else:
        if box[0]-border < 0:
            pad = 0-(box[0]-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+border > img_h:
            pad = (box[2]+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-border < 0:
            pad = 0-(box[1]-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+border > img_w:
            pad = (box[3]+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-border:box[2]+border, box[1]-border:box[3]+border]
    return img

def normalize(img, norm_range, min_val=None, max_val=None):
    if not min_val:
        min_val = img.min()

    if not max_val:
        max_val = img.max()

    img = (img - min_val) / (max_val - min_val)
    if norm_range == '-1,1':
        img = (img - 0.5) * 2  # Adjusts to -1 to 1 if desired
    return img

def loader(img, img_size, box=None, norm_range='0,1'):

    # Img
    min_val, max_val = img.min(), img.max()

    # Select Box Area
    if box:
        img = get_box(img, box, perc_border=0.5)

    # Resize
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)

    # Normalize
    img = normalize(img,  norm_range=norm_range, min_val=min_val, max_val=max_val)

    # To Tensor
    img = torch.Tensor(img)
    img = torch.unsqueeze(img, dim=0) # add dimension

    return img

class PrototypeDataset(torch.utils.data.Dataset):
    def __init__(self, data, opt, box_name='max_body_bbox'):
        """Initialize this dataset class.
        Parameters:
        """


        if opt['box_name']:
            box_data = pd.read_excel(opt['box_file'], index_col="id", dtype=list)
            self.boxes = {row[0]: eval(row[1][box_name]) for row in box_data.iterrows()}
        else:
            self.boxes = None




        # Check zipfile.
        self._zipfile = None
        if self._file_ext(self._path) == ".zip":
            self._type = "zip"
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a directory or zip")

        # Get the image paths.
        self.img_paths = sorted(fname for fname in self._all_fnames if self._file_ext(fname) == ".pickle" and  opt.phase in fname)
        if len(self.img_paths) == 0:
            raise IOError("No image files found in the specified path")

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index a random integer for data indexing
        """
        # read a image given a random integer index



        pass

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)

    # Added functions to manage zip file
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        else:
            raise IOError("Support only zip.")

# main
if __name__ == '__main__':
    pass