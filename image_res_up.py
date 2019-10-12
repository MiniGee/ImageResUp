
import numpy as np
from PIL import Image
from tqdm import tqdm

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from ModelBase.model_base import ModelBase


class ImageResUp(ModelBase):

    def __init__(self, name, loader):
        super(ImageResUp, self).__init__(name, loader)


    def create(self):
        input = Input(shape = (self._loader._lowres_size, self._loader._lowres_size, 3))

        x = Conv2DTranspose(32, 3, strides = 1, padding = 'same')(input)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2DTranspose(32, 3, strides = 1, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(16, 3, strides = 1, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2DTranspose(16, 3, strides = 1, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = Dense(16, activation = 'relu')(x)

        output = Dense(3, activation = 'sigmoid')(x)


        self._model = Model(inputs = input, outputs = output)
        self._model.summary()


    def compile(self):
        self._model.compile(
            optimizer = 'adam',
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )

        self._metrics = ['Loss', 'Accu']
        print(self._model.metrics_names)


    def generate(self, mb_size):
        features = self._loader.get_testing_batch(mb_size)[0]

        result = self._model.predict(features)
        result = (result * 255).astype(np.uint8)

        for i, data in enumerate(tqdm(result)):
            Image.fromarray(data).save('output/%d.png' % i)