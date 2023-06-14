import matplotlib.pyplot as plt
import tensorflow as tf

from settings.config import Config

def dice_coeff(y_true, y_pred):
    """Compute the Dice coefficient between two binary tensors.
    
    The Dice coefficient is a common evaluation metric used in image segmentation tasks.
    It measures the similarity between the predicted binary mask (y_pred) and the ground truth binary mask (y_true).

    Args:
        y_true: Tensor, the ground truth binary mask with values 0 or 1.
        y_pred: Tensor, the predicted binary mask with values 0 or 1.

    Returns:
        dice: Tensor, the Dice coefficient value between y_true and y_pred.
    """
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice


def BCE_Dice_loss(y_true, y_pred, alpha=Config.loss_alpha):
    """Compute the combined Binary Cross Entropy (BCE) and Dice loss.

    The BCE-Dice loss is a commonly used loss function in image segmentation tasks.
    It combines the Binary Cross Entropy loss and the Dice loss to optimize the model for both accuracy and overlap.

    Args:
        y_true: Tensor, the ground truth binary mask with values 0 or 1.
        y_pred: Tensor, the predicted logits or probabilities with the same shape as y_true.
        alpha: Float, the weight parameter for balancing the BCE and Dice components of the loss.

    Returns:
        loss: Tensor, the computed BCE-Dice loss.
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    loss = alpha * bce(y_true, y_pred) + (1 - alpha) * (1 - dice_coeff(y_true, y_pred))
    
    return loss


def f2_score(y_true, y_pred, beta=2):
    """Compute the F2 score, a metric commonly used in binary classification tasks.

    The F2 score is a variation of the F1 score that emphasizes recall over precision.
    It is useful when the imbalance between the positive and negative classes is high.

    Args:
        y_true: Tensor, the ground truth binary labels with values 0 or 1.
        y_pred: Tensor, the predicted probabilities or logits with the same shape as y_true.

    Returns:
        f2_score: Tensor, the computed F2 score.
    """
    y_pred = tf.cast(y_pred >= Config.threshold, y_pred.dtype)
    y_pred = tf.round(y_pred) 
    
    true_positives = tf.math.reduce_sum(y_true * y_pred)
    predicted_positives = tf.math.reduce_sum(y_pred)
    actual_positives = tf.math.reduce_sum(y_true)
    
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
    
    f2_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + tf.keras.backend.epsilon())
    
    return f2_score


def vis_logs(logs):
    """Visualize training logs including loss, dice coefficient, F2 score, and learning rate.

    Args:
        logs: Dict, a dictionary containing training logs including loss, dice coefficient,
              F2 score, and learning rate.

    Returns:
        None
    """
    plt.figure(figsize=(20,10))
    ax = plt.subplot(2, 2, 1)
    ax.plot(logs['loss'], label='train')
    ax.plot(logs['val_loss'], label='valid')
    ax.set_title('Loss')
    plt.legend()
    
    ax = plt.subplot(2, 2, 2)
    ax.plot(logs['dice_coeff'], label='train')
    ax.plot(logs['val_dice_coeff'], label='valid')
    ax.set_title('Dice coefficient')
    plt.legend()
    
    ax = plt.subplot(2, 2, 3)
    ax.plot(logs['f2_score'], label='train')
    ax.plot(logs['val_f2_score'], label='valid')
    ax.set_title('F2 score')
    plt.legend()
    
    ax = plt.subplot(2, 2, 4)
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    ax.plot(logs['lr'])
    ax.set_title('Learning Rate')
    
    plt.show()