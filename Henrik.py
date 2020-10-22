
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

print("running")


def main():
    #X, y = load_data()
    run()

    #printen er svaret til bildet
    #print(y[15000])
    #show_img(X[15000])


def show_img(img):
    plt.imshow(img, cmap="gray")
    plt.show()


def load_data_2d():
    x_data = np.array(pd.read_csv("handwritten_digits_images.csv"))
    y_data = np.array(pd.read_csv("handwritten_digits_labels.csv"))
    return x_data, y_data[:, 0]


def load_data():
    x_data = np.array(pd.read_csv("handwritten_digits_images.csv"))
    x_data = x_data.reshape(x_data.shape[0], 28, 28)
    y_data = np.array(pd.read_csv("handwritten_digits_labels.csv"))
    return x_data, y_data


def d3_to_2d(data):
    nsamples, nx, ny = data.shape
    return data.reshape((nsamples, nx * ny))


def run():
    X, y = load_data_2d()
    print("data loaded")

    numFolds = 5
    seed = 42
    num_estimators = 100


    MLP = MLPClassifier(solver='adam', early_stopping=True)
    MLP_bag = BaggingClassifier(base_estimator=MLPClassifier(solver='adam'), n_estimators=num_estimators, max_samples=1)#0.4/num_estimators, n_jobs=3)


    K_NN = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
    K_NN_bag = BaggingClassifier(base_estimator=K_NN, n_estimators=num_estimators, max_samples=1.0)#0.4/num_estimators, n_jobs=3)


    SVM = svm.SVC(gamma='scale', kernel='rbf')
    SVM_bag = BaggingClassifier(base_estimator=SVM, n_estimators=num_estimators, max_samples=1)#0.4/num_estimators, n_jobs=3)


    random_forest_gini = RandomForestClassifier(n_estimators=num_estimators, criterion='gini', n_jobs=3)
    random_forest_entropy = RandomForestClassifier(n_estimators=num_estimators, criterion='entropy', n_jobs=3)



    models = [
              ("MLP", MLP),
              ("MLP bag", MLP_bag),
              ("K-NN", K_NN), #K-NN er tar for lang tid med store datasett
              ("K-NN bag", K_NN_bag),
              ("SVM", SVM),
              ("SVM bag", SVM_bag),
              ("RF gini", random_forest_gini),
              ("RF entropy", random_forest_entropy),
            ]
    best_idx = KFold_model_selection(X, y, models, numFolds, seed)
    print("Best is: " + models[best_idx][0])




### Split+shuffle X and Y into k=num_folds different folds:
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


### Select a model via Kfold cross-validation:
def KFold_model_selection(X, Y, models, num_folds, seed):
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.3)
    X_train_folds, X_val_folds, Y_train_folds, Y_val_folds = KFold_split(X_train_val, Y_train_val, num_folds, seed)
    mean_val_scores = []
    for model in models:
        mean_val_score, val_fold_scores = perform_KFold_CV(X_train_folds, X_val_folds, Y_train_folds, Y_val_folds, model)
        print("Mean validation score for %s: %r" % (model[0], mean_val_score))
        mean_val_scores.append(mean_val_score)

        # plotting
        plot(model[0], val_fold_scores, "Folds", "Score")


    best_instance_idx = mean_val_scores.index(max(mean_val_scores))
    print("\n\nBest instance idx:", best_instance_idx)
    plot("best model", mean_val_scores, "Score", "Classifier", [models[0][0], models[1][0], models[2][0], models[3][0], models[4][0], models[5][0], models[6][0], models[7][0]])

    return best_instance_idx


### KFold cross-validation of a model:
def perform_KFold_CV(X_train_folds, X_val_folds, Y_train_folds, Y_val_folds, model):
    val_fold_scores = []
    cmpt = 0
    for X_train_fold, X_val_fold, Y_train_fold, Y_val_fold in zip(X_train_folds, X_val_folds, Y_train_folds, Y_val_folds):
        val_fold_score = assess_model(X_train_fold, X_val_fold, Y_train_fold, Y_val_fold, model)
        cmpt += 1
        print(model[0], "fold", str(cmpt) + "/" + str(len(X_val_folds)), "validation score:", val_fold_score)
        val_fold_scores.append(val_fold_score)
    mean_val_MSE = np.mean(val_fold_scores)

    return mean_val_MSE, val_fold_scores


def assess_model(X_train, X_test, Y_train, Y_test, model):
    model[1].fit(X_train, Y_train)

    Y_test_pred = model[1].predict(X_test)
    test_score = accuracy_score(Y_test, Y_test_pred)
    return test_score




def plot(title, scores, x_label, y_label, names=[1, 2, 3, 4, 5]):
    plt.figure()
    plt.title(title)

    plt.xlabel(x_label)
    plt.ylabel(y_label)


    plt.bar(np.linspace(min(scores), len(scores), len(scores)), scores, tick_label=names,
            width=0.5, color=['red', 'green', 'blue'])

    plt.show()


main()