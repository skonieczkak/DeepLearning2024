import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU, BatchNormalization, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def generate_model1(input_shape, 
                    num_classes, 
                    learning_rate=0.001,
                    dropout_rate1=0.25,
                    dropout_rate2=0.5,
                    weight_decay=0.0001,
                    decay_steps=100000,
                    decay_rate=0.96):
    """
    Generate Keras Sequential model according to proposed architecture (1).

    Args:
        learning_rate (float, optional): Learning rate of the neural network. Defaults to LR.

    Returns:
        Keras Sequential model, complied, ready to use (to call .fit method).
    """

    model = Sequential([
        Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), input_shape=input_shape),
        LeakyReLU(0.1),
        Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)),
        LeakyReLU(0.1),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate1),
        Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)),
        LeakyReLU(0.1),
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)),
        LeakyReLU(0.1),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate1),
        Flatten(),
        Dense(256, kernel_regularizer=l2(weight_decay)),
        LeakyReLU(0.1),
        Dropout(dropout_rate2),
        Dense(num_classes, activation='softmax')
        ], name='Custom_model_1')

    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)

    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy'])
    return model



def generate_model2(input_shape, 
                    num_classes, 
                    learning_rate=0.001,
                    dropout_rate1=0.2,
                    dropout_rate2=0.2,
                    dropout_rate3=0.2,
                    decay_steps=100000,
                    decay_rate=0.96):
    """
    Generate Keras Sequential model according to proposed architecture (3), allowing customization of key hyperparameters.

    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of output classes.
        learning_rate (float, optional): Initial learning rate for the optimizer. Defaults to 0.001.
        dropout_rate1 (float, optional): Dropout rate after the first dense layer. Defaults to 0.2.
        dropout_rate2 (float, optional): Dropout rate after the second dense layer. Defaults to 0.2.
        dropout_rate3 (float, optional): Dropout rate after the third dense layer. Defaults to 0.2.
        decay_steps (int, optional): Number of steps before the learning rate decay applies. Defaults to 100000.
        decay_rate (float, optional): Rate of learning rate decay. Defaults to 0.96.

    Returns:
        Keras Sequential model: Compiled model ready for training.
    """

    model = Sequential([
        Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=input_shape),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(32, kernel_size=(4, 4), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate1),
        Dense(64, activation='relu'),
        Dropout(dropout_rate2),
        Dense(32, activation='relu'),
        Dropout(dropout_rate3),
        Dense(num_classes, activation='softmax')
        ], name='Custom_model_2')

    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)

    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model