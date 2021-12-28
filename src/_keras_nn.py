import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD

from sklearn.metrics import accuracy_score

def predict(nn, X, label):
    # function to predict from one hot encodeded probs matrix -> labels
    probs = nn(X)
    preds = np.array([label[pred] for pred in np.argmax(probs, axis=1)])

    return preds

def main():
    # load and split data
    train = np.loadtxt('./data/raw/df_train.csv', skiprows=1, delimiter=',')
    test = np.loadtxt('./data/raw/df_test.csv', skiprows=1, delimiter=',')

    X_train, y_train = train[:, :-1], train[:, -1].astype(int)
    X_test, y_test = test[:, :-1], test[:, -1].astype(int)

    labels = np.unique(y_train)
    intcode = {labels[k]:k for k in range(len(labels))}
    label = {k:labels[k] for k in range(len(labels))}

    train_n = len(X_train)
    test_n = len(X_test)

    # scale feature matrix
    X = np.vstack((X_train, X_test))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test = X[:train_n, :], X[train_n:, :]

    # one hot encode target vector 
    encoder = OneHotEncoder(sparse=False)
    y_train_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_hot = encoder.fit_transform(y_test.reshape(-1, 1))


    #-------------------
    # construct model
    nn = Sequential()
    nn.add(Dense(30, input_shape=(9,), activation='relu', name='fc1'))
    nn.add(Dense(6, activation='softmax', name='output'))

    # Adam optimizer with learning rate of 0.001
    nn.compile(
            optimizer=SGD(lr=0.1),
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

    print('Neural Network Model Summary: ')
    print(nn.summary())

    # Train the model
    print('Starting Training')
    history = nn.fit(X_train, y_train_hot,
               batch_size=1,
               epochs=300,
               verbose=2)
               #validation_data=(X_test, y_test_hot))


    train_preds = predict(nn, X_train, label)
    test_preds = predict(nn, X_test, label)

    """
    print(train_preds, y_train)
    print(test_preds, y_test)

    print(accuracy_score(y_train, train_preds))
    print(accuracy_score(y_test, test_preds))
    """

    # pred = np.argmax(nn.predict(x), axis=1)
    train_res = nn.evaluate(X_train, y_train_hot, verbose=0)
    test_res = nn.evaluate(X_test, y_test_hot, verbose=0)


    #print(f'Final Train Loss: {train_res[0]}')
    print(f'\nFinal Train Accuracy: {train_res[1]}')
    #print(f'Final Test Loss: {test_res[0]}')
    print(f'Final Test Accuracy: {test_res[1]}\n')

if __name__ == '__main__':
    main()
