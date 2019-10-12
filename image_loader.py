
import os
from glob import glob
from tqdm import tqdm

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ModelBase.loader_base import LoaderBase


class ImageLoader(LoaderBase):

    def __init__(self, data_dir, data_ext, dtype = np.float32):
        super(ImageLoader, self).__init__(data_dir, data_ext, dtype)

        self._img_size = 80
        self._lowres_size = 40


    def convert_to_raw(self, dir):
        print('Converting JPG to raw...')

        for i, fname in enumerate(tqdm(glob(os.path.join(dir, '*.jpg')))):
            try:
                image = Image.open(fname)
            except IOError:
                continue

            # If image too small, skip
            if image.size[0] < self._img_size:
                continue

            data = np.asarray(image.resize((self._img_size, self._img_size)), dtype = np.uint8)

            # Write to file
            with open(os.path.join(self._data_dir, '%d.dat' % i), 'wb') as f:
                f.write(data)


    def _load_file(self, fname):
        with open(fname, 'rb') as f:
            data = np.array(list(f.read()), dtype = np.uint8)
            data = data.reshape((self._img_size, self._img_size, 3))

        self._x_train.append(data)


    def _format_batch(self, idx, data_x, data_y):
        # The labels are full sized images
        labels = data_x[idx]

        # Features are downscaled version
        features = [np.asarray(Image.fromarray(data).resize((self._lowres_size, self._lowres_size))) for data in labels]
        features = np.array(features, dtype = np.float32) / 255.0

        labels = labels.astype(np.float32) / 255.0

        return (features, labels)
