import numpy as np
from matplotlib import pyplot as plt

from scripts.utils import get_data
from scripts.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD

np.random.seed(1)

def predict(nn, X, label):
    # function to predict from one hot encodeded probs matrix -> labels
    probs = nn(X)
    preds = np.array([label[pred] for pred in np.argmax(probs, axis=1)])

    return preds

def main():
    # ------ loading and preprocessing ------

    # load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(raw=False, scaled=True)

    # define mappings from target vector
    labels = np.unique(np.hstack((y_train, y_test)))
    intcode = {labels[k]:k for k in range(len(labels))}
    label = {k:labels[k] for k in range(len(labels))}

    # one hot encode target vector 
    encoder = OneHotEncoder(sparse=False)
    y_train_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_val_hot = encoder.fit_transform(y_val.reshape(-1, 1))
    y_test_hot = encoder.fit_transform(y_test.reshape(-1, 1))


    #------ constructing model ------
    
    # network architecture
    nn = Sequential()
    nn.add(Dense(20, input_shape=(9,), activation='relu', name='fc1'))
    nn.add(Dense(6, activation='softmax', name='output'))

    # define optimiser, loss and optimisation target
    nn.compile(
            optimizer=SGD(lr=0.1),
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

    # print overview of neural net architecture (number of (active) params to train per layer and in total)
    print(nn.summary())

    # train model
    epochs = 100
    history = nn.fit(X_train, y_train_hot,
               batch_size=1,
               epochs=epochs,
               verbose=2,
               validation_data=(X_val, y_val_hot)) # train with validation split (here test data)

    # plot training/ validation accuracy and loss history
    fig, ax = plt.subplots(ncols=2, figsize=(8,3))
    for i, title in zip(range(2), ['loss', 'accuracy']):
        ax[i].plot(list(range(1, epochs+1)), history.history[title], c='blue', label=f'Training {title.title()}')
        ax[i].plot(list(range(1, epochs+1)), history.history['val_'+title], c='red', label=f'Validation {title.title()}') 
        ax[i].set_title(f'History of {title.title()}')
        ax[i].set_xlabel('#Epochs')
        ax[i].legend(loc='best')
    plt.show()

    train_preds = predict(nn, X_train, label)
    test_preds = predict(nn, X_test, label)

    # pred = np.argmax(nn.predict(x), axis=1)
    train_res = nn.evaluate(X_train, y_train_hot, verbose=0)
    test_res = nn.evaluate(X_test, y_test_hot, verbose=0)


    print(f'\nFinal Train Loss: {train_res[0]}')
    print(f'Final Train Accuracy: {train_res[1]}\n')
    print(f'Final Test Loss: {test_res[0]}')
    print(f'Final Test Accuracy: {test_res[1]}\n')

if __name__ == '__main__':
    main()
