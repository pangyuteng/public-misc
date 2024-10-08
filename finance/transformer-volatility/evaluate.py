import numpy as np
from train import get_model, checkpoint_filepath
from sklearn.model_selection import train_test_split
from tensorflow import keras

if __name__ == "__main__":

    X = np.load("X.npy")
    Y = np.load("Y.npy")
    x_train, x_test, yv_train, yv_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(x_train.shape, x_test.shape, yv_train.shape, yv_test.shape)
        
    y_train = yv_train[:,:,0]
    v_train = yv_train[:,:,1]

    y_test = yv_test[:,:,0]
    v_test = yv_test[:,:,1]

    model = get_model()
    model.load_weights(checkpoint_filepath)
    model.compile(
        loss=["mse","mse"],
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mse"],
    )

    out = model.evaluate(x_train, [y_train,v_train], verbose=1)
    print("train",out)
    print(y_train.shape,'positive count',np.sum(y_train))
    out = model.evaluate(x_test, [y_test,v_test], verbose=1)
    print("test",out)
    print(y_test.shape,'positive count',np.sum(y_test))
