
import os

from image_loader import ImageLoader
from image_res_up import ImageResUp

import numpy as np


def main():
    needs_convert = not os.path.exists('data_raw')
    
    loader = ImageLoader('data_raw', 'dat', dtype = np.uint8)

    if needs_convert:
        loader.convert_to_raw('data_jpg')

    loader.load(0.1)
    
    model = ImageResUp('resup', loader)
    model.create()
    model.compile()
    model.train(10, 80, 10)
    model.generate(16)


if __name__ == '__main__':
    main()