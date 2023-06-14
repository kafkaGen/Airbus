import tensorflow as tf

from tensorflow.keras import layers
from settings.config import Config


class ResNetUnet():
    """ ResNetU-Net model for semantic segmentation.
        
        Args:
            num_classes (int): Number of output classes.
            channels (list): List of number of channels in each block.
            input_shape (tuple): Shape of the input tensor (H, W, C).
    """
    def __init__(self, num_classes=Config.num_classes, channels=Config.channels, 
                 input_shape=Config.input_shape):
        self.channels = channels
        self.num_classes = num_classes
        self.input_shape = input_shape
    
    def build_model(self):
        """ Build the ResNetU-Net model.
        
        Returns:
            tf.keras.Model: ResNetU-Net model.
        """
        inputs = layers.Input(self.input_shape)
        
        skip_connectins = [inputs]
        for i, filter in enumerate(self.channels[:-1]):
            if i == 0:
                x = self.input_block(skip_connectins[-1], filter)
            else:
                x = self.residual_downsample_block(skip_connectins[-1], filter)
            skip_connectins.append(x)
        
        x = self.conv_block(skip_connectins[-1], self.channels[-1])
        
        upsamples = [x]
        for filter in self.channels[:-1][::-1]:
            x = self.upsampling_block(upsamples[-1], skip_connectins.pop(-1), filter)
            upsamples.append(x)
            
        outputs = layers.Conv2D(self.num_classes, (1,1), padding="same", activation="sigmoid")(upsamples[-1])
        
        return tf.keras.Model(inputs, outputs, name="ResNetU-Net")
    
    def residual_downsample_block(self, x, filter):
        """ Residual downsample block is a component of the ResNetU-Net model. It is an 
        enhancement over the original architecture called ResNet-D, which addresses the issue 
        of information loss caused by double strides while maintaining the computational 
        efficiency.
        
        Args:
            x (tf.Tensor): Input tensor.
            filter (int): Number of filters in the convolutional layers.
            
        Returns:
            tf.Tensor: Output tensor.
        """
        # ResNet-D
        x_skip = x
        
        x = layers.Conv2D(filter, (1,1), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filter, (3,3), strides=2, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filter, (1,1), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        
        x_skip = layers.AveragePooling2D((2,2), strides=2, padding='same')(x_skip)
        x_skip = layers.Conv2D(filter, (1,1), padding='same', kernel_initializer='he_normal')(x_skip)
        x_skip = tf.keras.layers.BatchNormalization(axis=3)(x_skip)
        
        x = layers.add([x, x_skip])
        x = layers.ReLU()(x)
        
        return x
    
    def conv_block(self, x, filter):
        """ Convolutional block in the ResNetU-Net model.
        
        Args:
            x (tf.Tensor): Input tensor.
            filter (int): Number of filters in the convolutional layers.
            
        Returns:
            tf.Tensor: Output tensor.
        """
        x = layers.Conv2D(filter, (3,3), padding="same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filter, (3,3), padding="same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = layers.ReLU()(x)
        
        return x
    
    def upsampling_block(self, x, x_skip, filter):
        """ Upsampling block in the ResNetU-Net model.
        
        Args:
            x (tf.Tensor): Input tensor.
            x_skip (tf.Tensor): Skip connection tensor from the encoder.
            filter (int): Number of filters for the convolutional layers.
            
        Returns:
            tf.Tensor: Output tensor after applying the upsampling block.
        """
        x = layers.concatenate([x, x_skip])
        x = layers.Conv2DTranspose(filter, (3,3), strides=2, padding="same", kernel_initializer='he_normal')(x)
        x = self.conv_block(x, filter)
        
        return x
    
    def input_block(self, x, filter):
        """ Input block in the ResNetU-Net model.
        
        Args:
            x (tf.Tensor): Input tensor.
            filter (int): Number of filters for the convolutional layers.
            
        Returns:
            tf.Tensor: Output tensor after applying the input block.
        """
        x = self.conv_block(x, filter)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
        
        return x
    
    
class Unet():
    """ U-Net model for semantic segmentation.
        
        Args:
            num_classes (int): Number of output classes.
            channels (list): List of number of channels in each block.
            input_shape (tuple): Shape of the input tensor (H, W, C).
    """
    def __init__(self, num_classes=Config.num_classes, channels=Config.channels,
                 input_shape=Config.input_shape):
        self.channels = channels
        self.num_classes = num_classes
        self.input_shape = input_shape
    
    def build_model(self):
        """ Builds the U-Net model.

        Returns:
            tf.keras.Model: The U-Net model.
        """
        inputs = layers.Input(self.input_shape)
        
        skip_connectins = [inputs]
        for i, filter in enumerate(self.channels[:-1]):
            if i > 1:
                x = self.downsample_block(skip_connectins[-1], filter, p=0.2)
            x = self.downsample_block(skip_connectins[-1], filter, p=0)
            skip_connectins.append(x)
        
        x = self.conv_block(skip_connectins[-1], self.channels[-1], p=0.2)
        
        upsamples = [x]
        for filter in self.channels[:-1][::-1]:
            x = self.upsampling_block(upsamples[-1], skip_connectins.pop(-1), filter, p=0)
            upsamples.append(x)
            
        outputs = layers.Conv2D(self.num_classes, (1,1), padding="same", activation="sigmoid")(upsamples[-1])
        
        return tf.keras.Model(inputs, outputs, name="U-Net")
    
    def conv_block(self, x, filter, p):
        """ Creates a convolutional block.

        Args:
            x (tensor): Input tensor.
            filter (int): Number of filters in the block.

        Returns:
            tensor: Output tensor of the block.
        """
        x = layers.Conv2D(filter, (3,3), padding = "same", kernel_initializer="he_normal")(x) 
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(p)(x)
        x = layers.Conv2D(filter, (3,3), padding = "same", kernel_initializer="he_normal")(x) 
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = layers.ReLU()(x)
        
        return x
    
    def downsample_block(self, x, filter, p):
        """ Creates a downsample block.

        Args:
            x (tensor): Input tensor.
            filter (int): Number of filters in the block.

        Returns:
            tensor: Output tensor of the block.
        """
        x = self.conv_block(x, filter, p)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
        
        return x
    
    def upsampling_block(self, x, x_skip, filter, p):
        """ Creates an upsampling block.

        Args:
            x (tensor): Input tensor.
            x_skip (tensor): Skip connection tensor from the downsample block.
            filter (int): Number of filters in the block.

        Returns:
            tensor: Output tensor of the block.
        """
        x = layers.concatenate([x, x_skip])
        x = layers.Conv2DTranspose(filter, (3,3), strides=2, padding="same", kernel_initializer='he_normal')(x)
        x = self.conv_block(x, filter, p)
        
        return x
    
    
class ResNet():
    """ ResNet model.
    
        Args:
            num_classes (int): Number of output classes.
            channels (list): List of integers specifying the number of channels in each block.
            input_shape (tuple): Input shape of the model (height, width, channels).
    """
    def __init__(self, num_classes=Config.num_classes, channels=Config.channels, input_shape=Config.input_shape):
        self.channels = channels
        self.num_classes = num_classes
        self.input_shape = input_shape
    
    def build_model(self):
        """ Builds the ResNet model.

        Returns:
            tf.keras.Model: The constructed ResNet model.
        """
        inputs = layers.Input(self.input_shape)
        
        x = self.input_block(inputs, self.channels[0])
        
        for filter in self.channels:
            x = self.residual_block(x, filter)
            
        outputs = self.output_block(x)
        
        return tf.keras.Model(inputs, outputs, name="ResNet")
    
    def input_block(self, x, filters):
        """ Constructs the input block of the ResNet model.

        Args:
            x (tf.Tensor): Input tensor.
            filters (int): Number of filters/channels.

        Returns:
            tf.Tensor: Output tensor of the input block.
        """
        # ResNet-C
        x = layers.Conv2D(filters, (3,3), strides=2, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (3,3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (3,3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
        
        return x
    
    def residual_block(self, x, filter):
        """ Constructs a residual block of the ResNet model.

        Args:
            x (tf.Tensor): Input tensor.
            filter (int): Number of filters/channels.

        Returns:
            tf.Tensor: Output tensor of the residual block.
        """
        # ResNet-D
        x_skip = x
        
        x = layers.Conv2D(filter, (1,1), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filter, (3,3), strides=2, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filter, (1,1), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        
        x_skip = layers.AveragePooling2D((2,2), strides=2, padding='same')(x_skip)
        x_skip = layers.Conv2D(filter, (1,1), padding='same', kernel_initializer='he_normal')(x_skip)
        x_skip = tf.keras.layers.BatchNormalization(axis=3)(x_skip)
        
        x = layers.add([x, x_skip])
        x = layers.ReLU()(x)
        
        return x
    
    def output_block(self, x):
        """ Constructs the output block of the ResNet model.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor of the output block.
        """
        x = layers.GlobalMaxPooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(self.num_classes, activation='sigmoid')(x)
        
        return x