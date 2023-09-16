import pandas as pd

from automl.preprocess import build_classification_preprocessor
from automl.engine import AutoMLClassification


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X_train, y_train = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=5,
        n_redundant=20,
        n_repeated=20,
        n_clusters_per_class=2,
        n_classes=10,
        flip_y=0.1,
        class_sep=0.1,
        random_state=42,
    )

    X_train = pd.DataFrame(X_train)
    preprocesor = build_classification_preprocessor(X_train)

    auto_ml = AutoMLClassification(
        n_iter=10,
        cv=3,
        random_state=42,
        experiment_name="AutoML Classification",
    )
    auto_ml.fit(X_train, y_train)
