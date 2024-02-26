from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications import VGG16
import tensorflow as tf

def generate_resnet_model(input_shape, num_classes, learning_rate=0.001, decay_steps=100000, decay_rate=0.96):
    """
    Generate a Keras model using ResNet50 as the base for feature extraction.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of classes for the output layer.
        learning_rate (float, optional): Initial learning rate for the Adam optimizer. Defaults to 0.001.
        weight_decay (float, optional): L2 weight decay regularization factor. Defaults to 0.0001.
        decay_steps (int, optional): Number of steps before applying one learning rate decay. Defaults to 100000.
        decay_rate (float, optional): Learning rate decay rate. Defaults to 0.96.

    Returns:
        Keras model: Compiled model ready for training.
    """
    input_tensor = tf.keras.Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Ensuring the base model is not trainable
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(num_classes, activation='softmax')
    ])

    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule),
                  metrics=['accuracy'])
    
    return model



def generate_vgg_model(input_shape, num_classes, learning_rate=0.001, decay_steps=100000, decay_rate=0.96):
    """
    Generate a Keras model using VGG16 as the base for feature extraction.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of classes for the output layer.
        learning_rate (float, optional): Initial learning rate for the Adam optimizer. Defaults to 0.001.
        weight_decay (float, optional): L2 weight decay regularization factor. Defaults to 0.0001.
        decay_steps (int, optional): Number of steps before applying one learning rate decay. Defaults to 100000.
        decay_rate (float, optional): Learning rate decay rate. Defaults to 0.96.

    Returns:
        Keras model: Compiled model ready for training.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Ensuring the base model is not trainable
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(num_classes, activation='softmax')
    ])

    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule),
                  metrics=['accuracy'])
    
    return model