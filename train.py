import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from argparse import ArgumentParser

from src.dataset import AirbusDataset
from src.model import ResNet, ResNetUnet
from src.metrics import BCE_Dice_loss, f2_score, dice_coeff
from settings.config import Config

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--classificator_train', type=str,
                        default='y')
    parser.add_argument('--segmentator_train', type=str,
                        default='y')
    args = parser.parse_args()

    if args.classificator_train in ['y', 'yes']:
        
        print('#' * 25)
        print('Start of classificator train.')
        print('#' * 25)
        
        dataset = AirbusDataset()
        train_ds, valid_ds = dataset.get_datasets(segmentation=False)
        
        
        model = ResNet().build_model()
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Config.learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', f2_score])

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(Config.model_path, 'resnet.h5'),
            monitor='val_f2_score',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            initial_value_threshold=0.92
        )

        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_f2_score',
            factor=0.7,
            patience=3,
            mode='max',
            min_lr=1e-6
        )
        
        model.fit(train_ds, epochs=Config.epochs, validation_data=valid_ds,
                  callbacks=[checkpoint, lr_scheduler])
        
        print('#' * 25)
        print('End of classificator train.')
        print('#' * 25)

    if args.segmentator_train in ['y', 'yes']:
        
        print('#' * 25)
        print('Start of segmentator train.')
        print('#' * 25)
        
        dataset = AirbusDataset()
        train_ds, valid_ds = dataset.get_datasets(segmentation=True)

        model = ResNetUnet().build_model()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Config.learning_rate),
                        loss=BCE_Dice_loss,
                        metrics=[f2_score, dice_coeff])

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(Config.model_path, 'unet.h5'),
            monitor='val_dice_coeff',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            initial_value_threshold=0.75
        )

        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coeff',
            factor=0.7,
            patience=3,
            mode='max',
            min_lr=1e-6
        )

        model.fit(train_ds, epochs=Config.epochs, validation_data=valid_ds,
                callbacks=[checkpoint, lr_scheduler])
        
        print('#' * 25)
        print('End of segmentator train.')
        print('#' * 25) 
        