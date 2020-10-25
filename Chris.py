#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, mean_squared_error

from keras import backend
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Dropout


# Creates an image from the row in X_data
def create_image(X_data, row):
    X_data = X_data.reshape(X_data.shape[0], 28, 28)
    img = X_data[row]
    imgPlot = plt.imshow(img, cmap="Greys")
    plt.show()


# Split+shuffle X and Y into k=num_folds different folds:
def KFold_split(X, Y, num_folds, seed):
    KFold_splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    X_train_folds = []
    X_val_folds = []
    Y_train_folds = []
    Y_val_folds = []
    for (kth_fold_train_idxs, kth_fold_val_idxs) in KFold_splitter.split(X, Y):
        X_train_folds.append(X[kth_fold_train_idxs])
        X_val_folds.append(X[kth_fold_val_idxs])
        Y_train_folds.append(Y[kth_fold_train_idxs])
        Y_val_folds.append(Y[kth_fold_val_idxs])
    return X_train_folds, X_val_folds, Y_train_folds, Y_val_folds


# Select a model via Kfold cross-validation:
def KFold_model_selection(X, Y, models, num_folds, seed):
    # Extract a test set:
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
    # Extract train and validation folds:
    X_train_folds, X_val_folds, Y_train_folds, Y_val_folds = KFold_split(X_train_val, Y_train_val, num_folds, seed)

    mean_val_scores = []

    for model_name, model in models:
        print("\nNow preprocessing model: ", model_name)
        mean_val_score = perform_KFold_CV(X_train_folds, X_val_folds, Y_train_folds, Y_val_folds, model, model_name)
        print("Mean validation score:", mean_val_score)
        mean_val_scores.append(mean_val_score)

    best_instance_idx = mean_val_scores.index(max(mean_val_scores))
    best_model_instance = models[best_instance_idx]
    print("\n\nBest model instance:", best_model_instance[0])

    best_model_score = assess_model(X_train_val, X_test, Y_train_val, Y_test, best_model_instance[1],
                                    best_model_instance[0])
    print("Best model score:", best_model_score)


# KFold cross-validation of a model:
def perform_KFold_CV(X_train_folds, X_val_folds, Y_train_folds, Y_val_folds, model, model_name):
    val_fold_scores = []

    count = 1
    for X_train_fold, X_val_fold, Y_train_fold, Y_val_fold in zip(X_train_folds, X_val_folds, Y_train_folds,
                                                                  Y_val_folds):
        print("Processing fold ", count, "/5...")
        val_fold_score = assess_model(X_train_fold, X_val_fold, Y_train_fold, Y_val_fold, model, model_name)
        val_fold_scores.append(val_fold_score)
        count += 1

    mean_val_score = np.mean(val_fold_scores)
    return mean_val_score


# Fit and evaluate a model:
def assess_model(X_train, X_test, y_train, y_test, model, model_name):
    if model_name.startswith("MLP"):
        X_train, y_train = preprocess_data(X_train, y_train)
        X_test, y_test = preprocess_data(X_test, y_test)

        model.fit(X_train, y_train, batch_size=256, epochs=10, verbose=0)

        score = model.evaluate(X_test, y_test, verbose=0)[1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
    print(score)
    return score


# Scale the features(pixels), transform labels into a one-hot encoding vector (used for MLP)
def preprocess_data(X, y):
    return X / 255, to_categorical(y)


# Appends MLP models with different hyper parameters(dropouts and optimizers)
def append_MLP_to_models(models: list):
    # No dropout, optimizer = adam
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    models.append(("MLP classifier, no dropout, optimizer = adam", model))

    # Dropout = 0.25, optimizer = adam
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    models.append(("MLP classifier, dropout = 0.25, optimizer = adam", model))

    # Dropout = 0.5, optimizer = adam
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    models.append(("MLP classifier, dropout = 0.5, optimizer = adam", model))

    # No dropout, optimizer = SGD
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    models.append(("MLP classifier, no dropout, optimizer = SGD", model))

    # Dropout = 0.25, optimizer = SGD
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    models.append(("MLP classifier, dropout = 0.25, optimizer = SGD", model))

    # Dropout = 0.5, optimizer = SGD
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    models.append(("MLP classifier, dropout = 0.5, optimizer = SGD", model))

    return models


# Appends random forrest models with different hyper parameters(number of trees)
def append_rndforest_to_models(models: list, seed):
    for n_trees in [25, 50, 75, 100]:
        model = RandomForestClassifier(n_estimators=n_trees, random_state=seed)
        name = "Random forest classifier, number of trees = " + str(n_trees)
        models.append((name, model))
    return models


# Appends decision tree models with different hyper parameters(entropy/gini)
def append_dectree_to_models(models: list):
    model = DecisionTreeClassifier(criterion="gini")
    name = "Decision tree classifier, criterion = gini"
    models.append((name, model))

    model = DecisionTreeClassifier(criterion="entropy")
    name = "Decision tree classifier, criterion = entropy"
    models.append((name, model))

    return models


if __name__ == "__main__":
    X_path = "handwritten_digits_images.csv"
    y_path = "handwritten_digits_labels.csv"

    X_data = pd.read_csv(X_path, header=None)
    X_data = np.array(X_data)

    y_data = pd.read_csv(y_path, header=None)
    y_data = np.array(y_data[0])

    seed = 666

    models = list()
    models = append_dectree_to_models(models)
    models = append_rndforest_to_models(models, seed)
    models = append_MLP_to_models(models)

    num_folds = 5

    KFold_model_selection(X_data, y_data, models, num_folds, seed)