import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from argparse import ArgumentParser

from settings.config import Config
from src.utils import is_valid_directory, is_valid_file, test_parser, rle_encode
from src.model import ResNet, ResNetUnet

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--classificator_path', type=str,
                        default=os.path.join(Config.model_path, 'resnet.h5'))
    parser.add_argument('--segmentator_path', type=str,
                        default=os.path.join(Config.model_path, 'unet.h5'))
    parser.add_argument('--test_dir_path', type=str,
                        default=Config.test_path)
    parser.add_argument('--submission_dir_path', type=str,
                        default='./')
    args = parser.parse_args()
    
    # validate input parameters
    is_valid_file(parser, args.classificator_path)
    is_valid_file(parser, args.segmentator_path)
    is_valid_directory(parser, args.test_dir_path)
    is_valid_directory(parser, args.submission_dir_path)
    
    # create test dataset
    dataset = tf.data.Dataset.from_tensor_slices((os.listdir(args.test_dir_path)))
    dataset = dataset.map(test_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(Config.batch_size)
    dataset = dataset.prefetch(1)
    
    # segment prediction
    unet = ResNetUnet().build_model()
    unet.load_weights(args.segmentator_path)
    segmentation_pred = unet.predict(dataset.take(20))
    segmentation_pred = tf.cast(segmentation_pred >= Config.threshold, tf.float32)
    segmentation_pred = tf.round(segmentation_pred)
    
    # classification prediction
    resnet = ResNet().build_model()
    resnet.load_weights(args.classificator_path)
    classification_pred = resnet.predict(dataset.take(20))
    classification_pred = classification_pred.flatten()
    
    # prepare submition
    submission = {}
    for image_filename, label, mask in zip(os.listdir(args.test_dir_path), classification_pred, segmentation_pred):
        if label:
            submission[image_filename] = rle_encode(mask.numpy())
        else:
            submission[image_filename] = ''
            
    submission = pd.DataFrame().from_dict(submission, orient='index', columns=['EncodedPixels'])
    submission.index.name = 'ImageId'
    submission.reset_index().to_csv(os.path.join(args.submission_dir_path, 'submission.csv'), 
                                    index=False)
    