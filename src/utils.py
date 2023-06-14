import os
import numpy as np
import tensorflow as tf
import cv2 as cv

from settings.config import Config

def is_valid_directory(parser, path):
    """
    Check if the provided path is a valid directory.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path is a valid directory, False otherwise.
    """
    if not  os.path.exists(path):
        parser.error(f"Error: Directory '{path}' does not exist.")

    if not os.path.isdir(path):
        parser.error(f"Error: '{path}' is not a directory.")
        

def is_valid_file(parser, path):
    """
    Check if the provided path is a valid file.

    Args:
        parser (argparse.ArgumentParser): The argparse parser object.
        path (str): The path to check.

    Returns:
        bool: True if the path is a valid file, False otherwise.
    """
    if not os.path.exists(path):
        parser.error(f"Error: File '{path}' does not exist.")

    if not os.path.isfile(path):
        parser.error(f"Error: '{path}' is not a file.")
        
        
def test_parser(image_filename):
    """Parse the image for the test dataset.

    Args:
        image_filename: str, filename of the image.

    Returns:
        images: tf.Tensor, parsed image.
    """
    def read(image_filename):
        image_filename = image_filename.decode('utf-8')
        
        image = cv.imread(os.path.join(Config.test_path, image_filename)).astype(np.float32)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image / 255.0
            
        return image
    
    images = tf.numpy_function(read, [image_filename], tf.float32)

    return images


def rle_encode(mask):
    """Convert a mask image to run-length encoded (RLE) code.

    Args:
        mask: np.ndarray, mask image.

    Returns:
        rle_code: str, run-length encoded (RLE) code.
    """  
    # Transpose mask and flatten it
    mask_flattened = mask.T.flatten()
    # Pad the mask
    mask_padded = np.pad(mask_flattened, pad_width=(1, 1), mode='constant')

    # Pad the mask
    starts = np.where(mask_padded[1:] > mask_padded[:-1])[0] + 1
    # Pad the mask
    ends = np.where(mask_padded[:-1] > mask_padded[1:])[0] + 1

    lengths = ends - starts
    # Combine starts and lengths
    rle_pairs = np.column_stack((starts, lengths))  

    rle_code = ''
    for pair in rle_pairs:
        # Construct the RLE code
        rle_code += f'{pair[0]} {pair[1]} '  

    return rle_code.strip()