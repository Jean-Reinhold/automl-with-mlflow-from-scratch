import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron
from lightgbm import LGBMClassifier

classification_models = {
    "RandomForestClassifier": RandomForestClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "SVC": SVC(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(),
    "GaussianNB": GaussianNB(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "XGBClassifier": XGBClassifier(),
    "ExtraTreesClassifier": ExtraTreesClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "RidgeClassifier": RidgeClassifier(),
    "SGDClassifier": SGDClassifier(),
    "Perceptron": Perceptron(),
    "LGBMClassifier": LGBMClassifier(),
}

classification_param_distributions = {
    "RandomForestClassifier": {
        "n_estimators": np.arange(10, 200),
        "max_features": ["sqrt", "log2"],
        "max_depth": np.arange(1, 20),
        "min_samples_split": np.arange(2, 20),
        "min_samples_leaf": np.arange(1, 20),
    },
    "GradientBoostingClassifier": {
        "n_estimators": np.arange(10, 200),
        "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
        "max_depth": np.arange(1, 20),
    },
    "SVC": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto"],
        "kernel": ["linear", "rbf", "poly", "sigmoid", "gaussian"],
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10, 100],
        "penalty": [None, "l2"],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [(10,), (50,), (100,)],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
    },
    "GaussianNB": {},
    "DecisionTreeClassifier": {
        "max_depth": np.arange(1, 20),
        "min_samples_split": np.arange(2, 20),
        "min_samples_leaf": np.arange(1, 20),
    },
    "XGBClassifier": {
        "n_estimators": np.arange(10, 200),
        "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
        "max_depth": np.arange(1, 20),
    },
    "ExtraTreesClassifier": {
        "n_estimators": np.arange(10, 200),
        "max_features": ["sqrt", "log2"],
        "max_depth": np.arange(1, 20),
        "min_samples_split": np.arange(2, 20),
        "min_samples_leaf": np.arange(1, 20),
    },
    "AdaBoostClassifier": {
        "n_estimators": np.arange(10, 200),
        "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
    },
    "LinearDiscriminantAnalysis": {
        "solver": ["svd", "lsqr", "eigen"],
    },
    "QuadraticDiscriminantAnalysis": {},
    "RidgeClassifier": {
        "alpha": [0.1, 0.5, 1.0, 2.0],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
    },
    "SGDClassifier": {
        "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        "penalty": [None, "l2", "l1", "elasticnet"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    },
    "Perceptron": {
        "penalty": [None, "l2", "l1", "elasticnet"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "fit_intercept": [True, False],
    },
    "LGBMClassifier": {
        "num_leaves": np.arange(10, 100),
        "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
        "n_estimators": np.arange(10, 200),
    },
}
