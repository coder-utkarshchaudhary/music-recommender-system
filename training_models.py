import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

INPUT_PATH = r'spectral_feature_for_training'
OUTPUT_PATH = r'models'

class CustomDataset(tf.keras.preprocessing.image.ImageDataGenerator):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    
    def flow(self, x, y, shuffle=True):
        batches = super().flow(x, y, batch_size=self.batch_size, shuffle=shuffle)

        while True:
            batch_x, batch_y = next(batches)
            yield batch_x, batch_y

create_generator = CustomDataset(10)

def cnn_model(input_shape=(128,128,3), activate_extension=False):
    _input = tf.keras.Input(shape=input_shape)

    # We have more mfccs for a song. So we need to make the model a bit more deep for mfccs.
    if activate_extension:
        x = layers.Conv2D(128, (3,3), activation=tf.nn.gelu, use_bias=True, kernel_regularizer='l2')(_input)
        x = layers.MaxPool2D(pool_size=(2,2))(x)
        x = layers.Dropout(0.3)(x)
    else:
        x = _input

    x = layers.Conv2D(64, (3,3), activation=tf.nn.gelu, use_bias=True, kernel_regularizer='l1')(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(32, (3,3), activation=tf.nn.gelu, use_bias=True, kernel_regularizer='l1')(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(16, (3,3), activation=tf.nn.gelu, use_bias=True, kernel_regularizer='l1')(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation=tf.nn.relu)(x)
    x = layers.Dense(64, activation=tf.nn.relu)(x)
    x = layers.Dense(32, activation=tf.nn.relu)(x)
    _out = layers.Dense(10, activation=tf.nn.gelu)(x)

    model = models.Model(_input, _out)
    return model

def plot_graphs(model_history):
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.plot(model_history.history['loss'], label='Train')
    plt.plot(model_history.history['val_loss'], label='Validation')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(model_history.history['mse'], label='Train')
    plt.plot(model_history.history['val_mse'], label='Validation')
    plt.title('Metric - MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.show()

def train_for_mfccs(X_mfcc, y_mfcc, output_path):
    X_train, X_valid, y_train, y_valid = train_test_split(X_mfcc, y_mfcc, test_size=0.1, random_state=69)
    train = create_generator.flow(X_train, y_train)
    valid = create_generator.flow(X_valid, y_valid)

    model = cnn_model(activate_extension=True)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=3e-3), loss=tf.keras.losses.mae, metrics=['mse'])
    hist = model.fit(train, epochs=20, steps_per_epoch = X_train.shape[0]//10, validation_steps=X_valid.shape[0]//10,validation_data=valid, callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1, start_from_epoch=3))
    
    # Plot training curves
    plot_graphs(model_history=hist)
    model.save(os.path.join(output_path, f'mfcc_model.keras'))

def train_for_sc(X_sc, y_sc, output_path):
        X_train, X_valid, y_train, y_valid = train_test_split(X_sc, y_sc, test_size=0.1, random_state=69)
        train = create_generator.flow(X_train, y_train)
        valid = create_generator.flow(X_valid, y_valid)

        model = cnn_model(activate_extension=False)
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=3e-3), loss=tf.keras.losses.mae, metrics=['mse'])
        hist = model.fit(train, epochs=20, steps_per_epoch = X_train.shape[0]//10, validation_steps=X_valid.shape[0]//10,validation_data=valid, callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1, start_from_epoch=3))
        
        # Plot training curves
        plot_graphs(model_history=hist)
        model.save(os.path.join(output_path, f'sc_model.keras'))

def train_for_sb(X_sb, y_sb, output_path):
        X_train, X_valid, y_train, y_valid = train_test_split(X_sb, y_sb, test_size=0.1, random_state=69)
        train = create_generator.flow(X_train, y_train)
        valid = create_generator.flow(X_valid, y_valid)

        model = cnn_model(activate_extension=False)
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=3e-3), loss=tf.keras.losses.mae, metrics=['mse'])
        hist = model.fit(train, epochs=20, steps_per_epoch = X_train.shape[0]//10, validation_steps=X_valid.shape[0]//10,validation_data=valid, callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1, start_from_epoch=3))
        
        # Plot training curves
        plot_graphs(model_history=hist)
        model.save(os.path.join(output_path, f'sb_model.keras'))

def train_for_zcr(X_zcr, y_zcr, output_path):
        X_train, X_valid, y_train, y_valid = train_test_split(X_zcr, y_zcr, test_size=0.1, random_state=69)
        train = create_generator.flow(X_train, y_train)
        valid = create_generator.flow(X_valid, y_valid)

        model = cnn_model(activate_extension=False)
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=3e-3), loss=tf.keras.losses.mae, metrics=['mse'])
        hist = model.fit(train, epochs=20, steps_per_epoch = X_train.shape[0]//10, validation_steps=X_valid.shape[0]//10,validation_data=valid, callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1, start_from_epoch=3))
        
        # Plot training curves
        plot_graphs(model_history=hist)
        model.save(os.path.join(output_path, f'zcr_model.keras'))

def get_input_arrays(input_path, name):
    X = np.load(os.path.join(input_path, f'{name}_X.npy'))
    y = np.load(os.path.join(input_path, f'{name}_y.npy'))
    return X,y

def main(input_path, names=['mfcc', 'sc', 'sb', 'zcr']):
    train_for_mfccs(get_input_arrays(input_path, names[0])[0], get_input_arrays(input_path, names[0])[1], OUTPUT_PATH)
    train_for_sc(get_input_arrays(input_path, names[1])[0], get_input_arrays(input_path, names[1])[1], OUTPUT_PATH)
    train_for_sb(get_input_arrays(input_path, names[2])[0], get_input_arrays(input_path, names[2])[1], OUTPUT_PATH)
    train_for_zcr(get_input_arrays(input_path, names[3])[0], get_input_arrays(input_path, names[3])[1], OUTPUT_PATH)

if __name__ == "__main__":
    main(INPUT_PATH)