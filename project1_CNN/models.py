import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU, BatchNormalization, Add, LayerNormalization, GlobalAveragePooling2D, Activation
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

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

def custom_model1(shape, number_classes, learning_rate=0.001, dropout_rate1=0.25, dropout_rate2=0.5, weight_decay=0.0001, decay_steps=100000, decay_rate=0.96):
    inputs = Input(shape=shape)
    
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
    outputs = Dense(number_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Advanced_Custom_Model_1')

    learning_schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    optimizer = Adam(learning_rate=learning_schedule)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def custom_model2(shape, number_classes, learning_rate=0.0001, dropout_rate=0.2, weight_decay=0.0003):
    inputs = Input(shape=shape)
    
    x = Conv2D(32, (3, 3), activation='relu', input_shape=shape)(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    x = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout_rate)(x)
    
    x = residual_attention_block(x, 64, weight_decay=weight_decay)
    x = residual_attention_block(x, 128, weight_decay=weight_decay)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(number_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Advanced_Custom_Model_2')

    learning_schedule = PiecewiseConstantDecay(list(range(0, 100, 10)), [learning_rate * 0.7 ** (i // 5) for i in range(11)])
    optimizer = Adam(learning_rate=learning_schedule)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def efficientnet_model(shape, 
                          number_classes, 
                          learning_rate=0.001):
    input_tensor = tf.keras.Input(shape=shape)
    architecture = EfficientNetB3(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Ensuring the base model is not trainable
    architecture.trainable = False

    model = Sequential([
        architecture,
        Flatten(),
        Dense(number_classes, activation='softmax')
    ], name='EfficientNet')

    learning_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.92,
        staircase=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=learning_schedule),
                  metrics=['accuracy'])
    
    return model