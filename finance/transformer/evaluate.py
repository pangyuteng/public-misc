import numpy as np
from train import get_model, checkpoint_filepath
from sklearn.model_selection import train_test_split
from tensorflow import keras

if __name__ == "__main__":

    X = np.load("X.npy")
    Y = np.load("Y.npy")
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    
    model = get_model()
    model.load_weights(checkpoint_filepath)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )
    out = model.evaluate(x_train, y_train, verbose=1)
    print("train",out)
    print(y_train.shape,'positive count',np.sum(y_train))
    out = model.evaluate(x_test, y_test, verbose=1)
    print("test",out)
    print(y_test.shape,'positive count',np.sum(y_test))
