# Torch DataLoader for RC dataset

# 2D # todo fruffini
# class ClaroDataset2D(Dataset):
#     def __init__(self, cfg, mode='train'):

#     def __getitem__(self, idx):

class ClaroDataset:
    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:

        """


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
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths

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