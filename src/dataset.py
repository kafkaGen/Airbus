import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2 as cv
import albumentations as A

from sklearn.model_selection import train_test_split

from settings.config import Config


class AirbusDataset():
    """Custom dataset class for the Airbus dataset.

    Args:
        images_path: str, path to the directory containing the images.
        masks_path: str, path to the CSV file containing the mask information.
        hue_path: str, path to the CSV file containing the hue information.
        labels_path: str, path to the CSV file containing the label information.
        batch_size: int, batch size for the dataset.
        buffer_size: int, buffer size for shuffling the dataset.
        random_state: int, random state for data splitting.

    Attributes:
        images_path: str, path to the directory containing the images.
        batch_size: int, batch size for the dataset.
        buffer_size: int, buffer size for shuffling the dataset.
        random_state: int, random state for data splitting.
        masks: pd.DataFrame, DataFrame containing the mask information.
        hue: pd.DataFrame, DataFrame containing the hue information.
        labels: pd.DataFrame, DataFrame containing the label information.
        train_seg_df: pd.DataFrame, DataFrame containing the training set mask information for segmentation.
        valid_seg_df: pd.DataFrame, DataFrame containing the validation set mask information for segmentation.
        train_cl_df: pd.DataFrame, DataFrame containing the training set label information for classification.
        valid_cl_df: pd.DataFrame, DataFrame containing the validation set label information for classification.

    Methods:
        __init__(self, images_path, masks_path, hue_path, labels_path, batch_size, buffer_size, random_state):
            Initialize the AirbusDataset object.
        rle_decode(self, mask_rle, shape=(768, 768)):
            Decode the run-length encoded mask.
        get_datasets(self, segmentation=True):
            Get the training and validation datasets.
        __segmentation_parser(self, image_filename, mask_filename):
            Parse the image and mask for the segmentation task.
        __classification_parser(self, image_filename, label_filename):
            Parse the image and label for the classification task.
        # __build_dataset(self, mask_df, parser, subset, shuffle=True):
            Build a TensorFlow dataset from the mask DataFrame.
        smart_crop(self, image, mask, padding_size=256):
            Perform smart cropping on the image and mask.
    """
    def __init__(self, images_path=Config.images_path, masks_path=Config.filtered_masks_path, 
                 hue_path=Config.hue_path, labels_path=Config.labels_path, 
                 batch_size=Config.batch_size, buffer_size=Config.buffer_size, 
                 random_state=Config.random_state):
        self.images_path = images_path
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.random_state = random_state
        
        self.masks = pd.read_csv(masks_path).set_index('ImageId')
        self.hue = pd.read_csv(hue_path).set_index('ImageId')
        self.labels = pd.read_csv(labels_path).set_index('ImageId')
        
        self.train_seg_df, self.valid_seg_df = train_test_split(self.masks, test_size=0.1, stratify=self.hue, random_state=self.random_state)
        self.train_cl_df, self.valid_cl_df = train_test_split(self.labels, test_size=0.1, random_state=self.random_state) 

    def rle_decode(self, mask_rle, shape=(768, 768)):
        """Decode the run-length encoded mask.

        Args:
            mask_rle: str, run-length encoded mask.
            shape: tuple, shape of the mask (default: (768, 768)).

        Returns:
            img: np.ndarray, decoded mask image.
        """
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T
    
    def get_datasets(self, segmentation=True):
        """Get the training and validation datasets.

        Args:
            segmentation: bool, flag indicating whether to retrieve datasets for segmentation (default: True).

        Returns:
            train_ds: tf.data.Dataset, training dataset.
            valid_ds: tf.data.Dataset, validation dataset.
        """
        if segmentation:
            train_ds = self.__build_dataset(self.train_seg_df, self.__segmentaion_parser, 'train')
            valid_ds = self.__build_dataset(self.valid_seg_df, self.__segmentaion_parser, 'valid', shuffle=False)
        
        else:
            train_ds = self.__build_classification_dataset(self.train_cl_df, self.__classification_parser, 'train')
            valid_ds = self.__build_classification_dataset(self.valid_cl_df, self.__classification_parser, 'valid', shuffle=False)
            
        return train_ds, valid_ds
            
    def __segmentaion_parser(self, image_filename, mask_filename):
        """Parse the image and mask for the segmentation task.

        Args:
            image_filename: str, filename of the image.
            mask_filename: str, filename of the image and rle code.

        Returns:
            images: tf.Tensor, parsed image.
            masks: tf.Tensor, parsed mask.
        """
        def read(image_filename, mask_filename, subset):
            image_filename = image_filename.decode('utf-8')
            mask_filename = [el.decode('utf-8') for el in mask_filename]
            
            image = cv.imread(image_filename).astype(np.float32)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = image / 255.0
            
            mask = self.rle_decode(mask_filename[1])
            mask = mask[:, :, np.newaxis].astype(np.float32)
            
            if subset.decode() == 'train':
                image, mask = self.smart_crop(image, mask)
                aug = get_transforms(subset.decode())(image=image, mask=mask)
                image, mask = aug['image'], aug['mask']
                
            return image, mask
        
        images, masks = tf.numpy_function(read, [image_filename, mask_filename, tf.constant(self.subset)], [tf.float32, tf.float32])

        return images, masks
    
    def __classification_parser(self, image_filename, labels):
        """Parse the image and label for the classification task.

        Args:
            image_filename: str, filename of the image.

        Returns:
            images: tf.Tensor, parsed image.
            labels: tf.Tensor, parsed label.
        """
        def read(image_filename, subset):
            image_filename = image_filename.decode('utf-8')
            
            image = cv.imread(os.path.join(Config.images_path, image_filename)).astype(np.float32)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = image / 255.0
            
            if subset.decode() == 'train':
                aug = get_transforms(subset.decode())(image=image)
                image = aug['image']
                
            return image
        
        images = tf.numpy_function(read, [image_filename, tf.constant(self.subset)], tf.float32)

        return images, labels

    def __build_dataset(self, mask_df, parser, subset, shuffle=True):
        """Build a TensorFlow dataset from the mask DataFrame.

        Args:
            mask_df: pd.DataFrame, DataFrame containing the mask information.
            parser: function, parser function to parse the image and mask.
            subset: str, subset identifier.
            shuffle: bool, flag indicating whether to shuffle the dataset (default: True).

        Returns:
            dataset: tf.data.Dataset, built TensorFlow dataset.
        """
        self.subset = subset
        
        image_paths = [os.path.join(Config.images_path, filename) for filename in mask_df.index]
        masks_paths = mask_df.astype(str).reset_index().to_numpy()

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, masks_paths))
        dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        
        return dataset
    
    
    def __build_classification_dataset(self, mask_df, parser, subset, shuffle=True):
        """Build a TensorFlow dataset from the mask DataFrame.

        Args:
            mask_df: pd.DataFrame, DataFrame containing the mask information.
            parser: function, parser function to parse the image and mask.
            subset: str, subset identifier.
            shuffle: bool, flag indicating whether to shuffle the dataset (default: True).

        Returns:
            dataset: tf.data.Dataset, built TensorFlow dataset.
        """
        self.subset = subset
        
        labels = tf.constant(mask_df.values.flatten(), dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((mask_df.index, labels))
        dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(2)
        
        return dataset
    
    def smart_crop(self, image, mask, padding_size=256):
        """Perform smart cropping on the image and mask. Method is to perform 
        intelligent cropping of an image and its corresponding mask by identifying
        and cropping around objects present in the mask. The goal is to focus on 
        the relevant objects and exclude any unnecessary background.

        Args:
            image: np.ndarray, image array.
            mask: np.ndarray, mask array.
            padding_size: int, size of the padded area (default: 256).

        Returns:
            image: np.ndarray, cropped and resized image.
            mask: np.ndarray, cropped and resized mask.
        """
        # Find contours of the objects in the mask
        mask = cv.convertScaleAbs(mask)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Create a bounding box around each object and calculate the minimum enclosing rectangle
        bounding_boxes = [cv.boundingRect(contour) for contour in contours]

        if len(bounding_boxes):
            cropped_images = []
            cropped_masks = []
            for (x, y, w, h) in bounding_boxes:
                # Crop the image based on the bounding box coordinates
                x1, y1, x2, y2 = y, x, y + h, x + w
                
                # Calculate object center
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Calculate the amount of padding required to reach the desired size
                pad_x = int(max((padding_size - h) / 2, 0))
                pad_y = int(max((padding_size - w) / 2, 0))
                
                # Calculate new point according to padding and image borders
                x1_pad, y1_pad = max(x1 - pad_x - max(x2 + pad_x - 768, 0), 0), max(y1 - pad_y - max(y2 + pad_y - 768, 0), 0)
                x2_pad, y2_pad = min(x2 + pad_x - min(x1 - pad_x, 0), 768), min(y2 + pad_y - min(y1 - pad_y, 0), 768)

                # Calculate the random shift for padding from the object using normal distribution
                shift_x = int(np.random.normal(loc=10, scale=pad_x * 2))
                shift_y = int(np.random.normal(loc=10, scale=pad_y * 2))
                
                # Calculate new point according to shift and image borders
                x1_shift, y1_shift = max(x1_pad - shift_x - max(x2_pad - shift_x - 768, 0), 0), max(y1_pad - shift_y - max(y2_pad - shift_y - 768, 0), 0)
                x2_shift, y2_shift = min(x2_pad - shift_x - min(x1_pad - shift_x, 0), 768), min(y2_pad - shift_y - min(y1_pad - shift_y, 0), 768)
                if x1_shift < center_x < x2_shift and y1_shift < center_y < y2_shift:
                    x1_pad, y1_pad = x1_shift, y1_shift
                    x2_pad, y2_pad = x2_shift, y2_shift
                
                cropped_images.append(image[x1_pad:x2_pad, y1_pad:y2_pad])
                cropped_masks.append(mask[x1_pad:x2_pad, y1_pad:y2_pad])
            
            # Find crop this the most information    
            idx = np.argmax([el.sum() for el in cropped_masks])
            
            image = cv.resize(cropped_images[idx], (256, 256))
            mask = cv.resize(cropped_masks[idx], (256, 256)).astype(np.float32)
        else:
            # if no object on mask - random crop
            x = np.random.randint(0, 756 - 256 + 1)
            y = np.random.randint(0, 756 - 256 + 1)
            image = image[y:y+256, x:x+256]
            mask = np.zeros((256, 256))

        return image, mask
    
    
def get_transforms(subset):
    """Get the augmentation transforms based on the subset.

    This function returns a composition of augmentation transforms based on the subset provided.

    Args:
        subset (str): The subset name, either 'train' or 'valid'.

    Returns:
        A.Compose: The composition of augmentation transforms.
    """
    if subset == 'train':
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.ColorJitter(brightness=[0.8, 1.2], saturation=[0.8, 1.2], hue=0.05),
                        A.RandomGamma(gamma_limit=[80, 120], eps=1e-4)
                    ]
                ),
                A.Flip(),
                A.Affine(rotate=(15, 45), mode=cv.BORDER_REFLECT),
                A.ChannelShuffle(),
                A.OneOf(
                    [
                        A.ElasticTransform(alpha=15, sigma=3, alpha_affine=5),
                        A.Blur(blur_limit=3),
                        A.GaussianBlur(blur_limit=3),
                        A.MedianBlur(blur_limit=3),
                    ]
                ),
            ]
        )
    else:
        return A.Compose(
            [
            ]
        )