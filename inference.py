import os
import filetype
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2 as cv

from argparse import ArgumentParser

from src.utils import is_valid_directory
from src.model import ResNet, ResNetUnet
from settings.config import Config

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='path to folder with images to segment')
    parser.add_argument('--output', type=str, required=True,
                        help='folder path to segmented images')
    args = parser.parse_args()
    
    is_valid_directory(parser, args.input)
    is_valid_directory(parser, args.output)
    
    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow has detected GPUs.")
    else:
        print("No GPUs found. TensorFlow is using CPU.")
    
    resnet = ResNet().build_model()
    resnet.load_weights(os.path.join(Config.model_path, 'resnet.h5'))
    
    unet = ResNetUnet().build_model()
    unet.load_weights(os.path.join(Config.model_path, 'unet.h5'))
    
    for el in os.listdir(args.input):
        fl_path = os.path.join(args.input, el)
        if not filetype.is_image(fl_path):
            continue
        
        image = cv.imread(fl_path).astype(np.float32)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)[np.newaxis, :, :, :]
        
        image = image / 255.0
        classification_pred = resnet(image).numpy().flatten()[0]
        # TODO pretrian classification model
        classification_pred = 1
        if classification_pred:
            print(f'{fl_path}: objects detected.')
            mask = unet(image)
            mask = tf.cast(mask >= Config.threshold, tf.float32)
            mask = tf.round(mask).numpy()[0, :, :, 0]
            
        else:
            print(f'{fl_path}: objects not detected.')
            mask = np.zeros(image.shape[1:3])
            
        cv.imwrite(os.path.join(args.output, el), mask * 255)
