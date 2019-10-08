
import numpy as np

from ModelBase.loader_base import LoaderBase


class ImageLoader(LoaderBase):

    def __init__(self, data_dir, dtype = np.float32):
        super(ImageLoader, self).__init__(data_dir, dtype)