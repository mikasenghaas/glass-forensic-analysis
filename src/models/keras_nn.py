import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
from matplotlib import pyplot as plt

from scripts.utils import get_data, generate_summary
from scripts.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD

np.random.seed(1)

SHOW = True
SAVE = True

def predict(nn, X, label):
    # function to predict from one hot encodeded probs matrix -> labels
    probs = nn(X)
    preds = np.array([label[pred] for pred in np.argmax(probs, axis=1)])

    return preds

def main():
    # ------ loading and preprocessing ------

    # load and split data
    X_train, X_test, y_train, y_test = get_data(raw=True, scaled=True)

    # define mappings from target vector
    labels = np.unique(np.hstack((y_train, y_test)))
    intcode = {labels[k]:k for k in range(len(labels))}
    label = {k:labels[k] for k in range(len(labels))}

    # one hot encode target vector 
    encoder = OneHotEncoder(sparse=False)
    y_train_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_hot = encoder.fit_transform(y_test.reshape(-1, 1))

    #------ constructing model ------
    
    # network architecture
    nn = Sequential()
    nn.add(Dense(50, input_dim=9, activation='relu', name='fc1'))
    #nn.add(keras.layers.Dropout(0.4))
    #nn.add(Dense(10, activation='relu', name='fc2'))
    #nn.add(keras.layers.Dropout(0.4))
    nn.add(Dense(6, activation='softmax', name='output'))

    # define optimiser, loss and optimisation target
    nn.compile(
            optimizer=Adam(lr=0.0005),
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

    # print overview of neural net architecture (number of (active) params to train per layer and in total)
    print(nn.summary())
    
    #es_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min')
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        
    # train model
    epochs = 500
    history = nn.fit(X_train, y_train_hot,
               batch_size=5,
               epochs=epochs,
               verbose=2,
               validation_split=0.2,
               callbacks=[es]) # train with validation split (here test data)

    # plot training/ validation accuracy and loss history
    fig, ax = plt.subplots(ncols=2, figsize=(8,3))
    for i, title in zip(range(2), ['loss', 'accuracy']):
        ax[i].plot(range(len(history.history[title])), history.history[title], label=f'Training {title.title()}')
        ax[i].plot(range(len(history.history['val_' + title])), history.history['val_'+title], label=f'Validation {title.title()}') 
        ax[i].set_title(f'History of {title.title()}')
        ax[i].set_xlabel('#Epochs')
        ax[i].legend(loc='best')

    if SHOW:
        print('show')
        plt.show()

        if input('SAVE? (y/n)' ) == 'y':
            fig.savefig('./data/figures/keras_nn_training.pdf')
            print('saved')

    # evaluate
    train_preds = predict(nn, X_train, label)
    test_preds = predict(nn, X_test, label)

    # pred = np.argmax(nn.predict(x), axis=1)
    train_res = nn.evaluate(X_train, y_train_hot, verbose=0)
    test_res = nn.evaluate(X_test, y_test_hot, verbose=0)

    train_loss = train_res[0]
    train_acc = train_res[1]
    test_loss = test_res[0]
    test_acc = test_res[1]

    conf_matrix = confusion_matrix(y_test, test_preds, as_frame=True, normalised=False)
    report = classification_report(y_test, test_preds)

    if SHOW:
        print(f'\nFinal Train Loss: {train_loss}')
        print(f'Final Train Accuracy: {train_acc}\n')
        print(f'Final Test Loss: {test_loss}')
        print(f'Final Test Accuracy: {test_acc}\n')

        print(conf_matrix)
        print(report)

        if input('SAVE? (y/n)' ) == 'y':
            generate_summary(filepath = './data/results', name='keras_neural_network', 
                             training_accuracy = train_acc,
                             training_loss = train_loss, 
                             test_accuracy = test_acc,
                             test_loss = test_loss,
                             confusion_matrix = conf_matrix,
                             classification_report = report)

if __name__ == '__main__':
    main()
