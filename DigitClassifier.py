import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC


# Loads in the data
def load_data(file):
    data = np.loadtxt(file, delimiter=',', dtype=np.int16)
    return data


# Splits the data into training, test and validation sets
def create_training_test_validation(x, y, random_state_nr=0):
    print("Creates training, test and validation sets")

    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4, random_state=random_state_nr)
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.5, random_state=random_state_nr)

    return x_train, y_train, x_test, y_test, x_val, y_val


# Returns a dict to preform a RandomizedSearchCV
def hyperparameter_mlp():
    hidden_layer_sizes = [(350,), (500,)]
    alpha = [0.0001, 0.05, 1e-04]
    learning_rate = ['constant', 'adaptive']

    param_grid = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": alpha,
        "learning_rate": learning_rate,
    }
    return param_grid


# Returns a dict to preform a RandomizedSearchCV
def hyperparameter_knn():
    # Neighbours to be tested
    n_neighbours = [1, 3, 5, 7, 9]
    # Weights to be used
    weights = ["uniform", "distance"]
    # Metrics to be used
    metric = ["euclidean", "manhattan"]
    param_grid = {"n_neighbors": n_neighbours, "weights": weights, "metric": metric}
    return param_grid


# Returns a dict to preform a RandomizedSearchCV
def hyperparameter_svc():
    # Different kernels to be tested
    kernel = ["linear", "rbf"]
    # Different regularization parameters
    c = [1, 10]
    # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    gamma = [0.01, 0.1, 1]
    param_grid = {"kernel": kernel, "C": c, "gamma": gamma}
    return param_grid


# Returns a dict to preform a RandomizedSearchCV
def hyperparameter_random_forest():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(50, 110, num=5)]
    max_depth.append(None)

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf}
    return param_grid


# Uses the parameter grid and does a randomized search for best hyper-parameters
# Also returns a model using these "best" parameters
def train_random_hyperparam(x_train, y_train, param_grid, model):
    print(f"Training the {type(model).__name__} model")

    # Random search of parameters, using 2 fold cross validation,
    # search across 10 different combinations
    model_random = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=2, verbose=1,
                                      random_state=0, n_jobs=-1)
    model_random.fit(x_train, y_train)

    print(f"The best params for the {type(model_random).__name__} model is {model_random.best_params_}")
    return model_random.best_estimator_


# Estimates how good the model preforms
def evaluate_model(x, y, model):
    y_pred = model.predict(x)
    score = metrics.r2_score(y, y_pred)
    print(f"The accuracy of the {type(model).__name__} model is {score:.4f}")
    return type(model).__name__, score


def train_grid_search_hyperparam(x_train, y_train, param_grid, model):
    print("Grid search")
    model_random = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
    model_random.fit(x_train, y_train)

    print(f"The best params for the {type(model_random).__name__} model is {model_random.best_params_}")
    return model_random.best_estimator_


def improve_mlp(x_train, y_train, x_test, y_test, x_val, y_val):
    print("Makes MLP for random")
    mlp_model = MLPClassifier(random_state=0, early_stopping=True)
    param_grid = hyperparameter_mlp()
    mlp_trained = train_random_hyperparam(x_train, y_train, param_grid, mlp_model)
    mlp_model_name, mlp_score = evaluate_model(x_val, y_val, mlp_trained)

    print("Makes MLP for grid")
    # mlp_model = MLPClassifier()
    # param_grid = hyperparameter_mlp_comprehensive()
    # mlp_trained = train_grid_search_hyperparam(x_train, y_train, param_grid, mlp_model)
    # mlp_model_name, mlp_score = evaluate_model(x_val, y_val, mlp_trained)


# Creates models, chooses hyper-parameters, estimates performance and announces best model.
def choose_model(x_train, y_train, x_test, y_test, x_val, y_val):
    print("------------------------------------------------------------------")
    print(f"----- Choose best model-----")

    print("-----MLP-----")
    # TODO hyper-parameters
    mlp_model = MLPClassifier(random_state=0, early_stopping=True)
    mlp_grid = hyperparameter_mlp()
    mlp_trained = train_random_hyperparam(x_train, y_train, mlp_grid, mlp_model)

    mlp_model_name, mlp_score = evaluate_model(x_val, y_val, mlp_trained)

    print("-----SVC-----")
    # TODO hyper-parameters
    svc_model = SVC()
    svc_grid = hyperparameter_svc()
    svc_trained = train_random_hyperparam(x_train, y_train, svc_grid, svc_model)

    # svc_trained = SVC(kernel='rbf', gamma=0.1, C=10)
    # svc_trained.fit(x_train, y_train)

    svc_model_name, svc_score = evaluate_model(x_val, y_val, svc_trained)

    print("-----Random Forest-----")
    # TODO hyper-parameters
    rf_model = RandomForestClassifier(random_state=0)
    rf_grid = hyperparameter_random_forest()
    rf_trained = train_random_hyperparam(x_train, y_train, rf_grid, rf_model)
    rf_model_name, rf_score = evaluate_model(x_val, y_val, rf_trained)

    # rf_trained = RandomForestClassifier(n_estimators=700, min_samples_leaf=4, max_features="sqrt", max_depth=65, random_state=0)
    # rf_trained.fit(x_train, y_train)

    rf_model_name, rf_score = evaluate_model(x_val, y_val, rf_trained)

    print("-----KNeighbors-----")
    # TODO hyper-parameters
    knn_model = KNeighborsClassifier()
    knn_grid = hyperparameter_knn()
    knn_trained = train_random_hyperparam(x_train, y_train, knn_grid, knn_model)

    # knn_trained = KNeighborsClassifier(weights="uniform", n_neighbors=7, metric="euclidean")
    # knn_trained.fit(x_train, y_train)

    knn_model_name, knn_score = evaluate_model(x_val, y_val, knn_trained)

    all_scores = {
        mlp_model_name: mlp_score,
        rf_model_name: rf_score,
        svc_model_name: svc_score,
        knn_model_name: knn_score
    }

    all_models = {
        mlp_model_name: mlp_trained,
        rf_model_name: rf_trained,
        svc_model_name: svc_trained,
        knn_model_name: knn_trained
    }

    print(f"----- Overview over all scores-----")
    [print(key, value) for key, value in all_scores.items()]
    print("------------------------------------------------------------------")

    model_with_best_accuracy = max(all_scores, key=all_scores.get)
    print(model_with_best_accuracy + " has the overall highest accuracy!")

    print("Will now use unseen data to estimate the expected performance:")
    evaluate_model(x_test, y_test, all_models.get(model_with_best_accuracy))
    print(f"Model selection is concluded-----")
    print("------------------------------------------------------------------")

    return all_models


def show_image(data, row):
    data = data.reshape(data.shape[0], 28, 28)
    img = data[row]
    matplotlib.pyplot.imshow(img, cmap="Greys")
    plt.show()


def preprocess(x):
    return x / 255


def main():
    X_file_name = "handwritten_digits_images.csv"
    y_file_name = "handwritten_digits_labels.csv"

    x_np = load_data(X_file_name)
    y_np = load_data(y_file_name)

    x_np = scale(x_np)
    x_train, y_train, x_test, y_test, x_val, y_val = create_training_test_validation(x_np, y_np)

    # mlp_model = MLPClassifier(alpha=0.0001, activation="relu", solver="adam", learning_rate="constant",
    #                           hidden_layer_sizes=(350,), random_state=0, early_stopping=True)
    # mlp_model.fit(x_train, y_train)
    # mlp_model_name, mlp_score = evaluate_model(x_val, y_val, mlp_model)
    # improve_mlp(x_train, y_train, x_test, y_test, x_val, y_val)

    choose_model(x_train, y_train, x_test, y_test, x_val, y_val)


main()