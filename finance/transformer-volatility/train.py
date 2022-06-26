"""
reference. https://keras.io/examples/timeseries/timeseries_transformer_classification
"""

import os
import random
import numpy as np
import tensorflow as tf
def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
SEED = 42069
set_seed(SEED)

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(
    input_shape,
    n_classes,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs0 = layers.Dense(2, activation="softmax")(x)
    outputs1 = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, [outputs0,outputs1])

checkpoint_filepath = 'model.h5'
def get_model():

    model = build_model(
        input_shape=(80,4),
        n_classes=2,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )
    return model

if __name__ == "__main__":

    X = np.load("X.npy")
    Y = np.load("Y.npy")
    x_train, x_test, yv_train, yv_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(x_train.shape, x_test.shape, yv_train.shape, yv_test.shape)
        
    idx = np.random.permutation(np.arange(x_train.shape[0]))

    x_train = x_train[idx,:,:]
    y_train = yv_train[idx,0]
    v_train = yv_train[idx,1]
    
    y_test = yv_test[:,0]
    v_test = yv_test[:,1]
    
    input_shape = x_train.shape[1:]
    assert(input_shape==(80,4))
    
    model = get_model()
    model.compile(
        loss=["sparse_categorical_crossentropy","sparse_categorical_crossentropy"],
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
            verbose=1, mode='min', min_delta=0.0001, cooldown=1, min_lr=0.00001),        
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        model_checkpoint_callback,
    ]

    model.fit(
        x_train,
        [y_train,v_train],
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
    )

    model.evaluate(x_test, [y_test,v_test], verbose=1)

