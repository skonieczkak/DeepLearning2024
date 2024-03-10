import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU, BatchNormalization, Add, LayerNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def conv_block(x, filters, kernel_size, strides=(1, 1), padding='same', activation='relu', weight_decay=0.0001):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=None, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_attention_block(input_tensor, filters, kernel_size=(3, 3), strides=(1, 1), weight_decay=0.0001):
    # Convolutional path
    x = conv_block(input_tensor, filters, kernel_size, strides=strides, weight_decay=weight_decay)
    x = conv_block(x, filters, kernel_size, weight_decay=weight_decay)
    x = conv_block(x, filters, kernel_size, weight_decay=weight_decay)
    
    # Residual path
    shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same', activation=None, kernel_regularizer=l2(weight_decay))(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = LayerNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def generate_model1(input_shape, num_classes, learning_rate=0.001, dropout_rate1=0.25, dropout_rate2=0.5, weight_decay=0.0001, decay_steps=100000, decay_rate=0.96):
    inputs = Input(shape=input_shape)
    
    x = conv_block(inputs, 32, (3, 3), weight_decay=weight_decay)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate1)(x)

    x = residual_attention_block(x, 64, weight_decay=weight_decay)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate1)(x)

    x = residual_attention_block(x, 128, weight_decay=weight_decay)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate2)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Advanced_Custom_Model_1')

    lr_schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_model2(input_shape, num_classes, learning_rate=0.001, dropout_rate1=0.2, dropout_rate2=0.2, dropout_rate3=0.2, decay_steps=100000, decay_rate=0.96):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (4, 4), activation='relu', input_shape=input_shape)(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    x = residual_attention_block(x, 64, weight_decay=0.0001)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate1)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate2)(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Advanced_Custom_Model_2')

    lr_schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model