import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import (
    make_scorer,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from automl.classification_search_space import (
    classification_param_distributions,
    classification_models,
)

mlflow.set_tracking_uri("http://0.0.0.0:5000")


class AutoMLClassification:
    def __init__(
        self,
        models: dict = classification_models,
        param_distributions: dict = classification_param_distributions,
        n_iter: int = 100,
        cv: int = 3,
        random_state: int = None,
        experiment_name: str = None,
        preprocessor: Pipeline = None,
    ):
        self.models = models
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.preprocessor = preprocessor
        self.experiment_name = f"{experiment_name} {int(datetime.now().timestamp())}"
        self.best_models_ = []

    def fit(self, X, y):
        experiment = mlflow.set_experiment(self.experiment_name)

        with ProcessPoolExecutor(max_workers=5) as executor:
            future_to_model = {
                executor.submit(
                    self.tune_model, X, y, model_name, model, experiment
                ): model_name
                for model_name, model in self.models.items()
            }
            self.best_models_ = []
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    best_model = future.result()
                    self.best_models_.append(best_model)
                except Exception as exc:
                    print(f"{model_name} generated an exception: {exc}")

        return self

    def tune_model(self, X, y, model_name: str, model, experiment):
        with mlflow.start_run(
            experiment_id=experiment.experiment_id, run_name=model_name
        ):
            pipeline = Pipeline([("preprocessor", self.preprocessor), ("model", model)])
            param_distributions = {
                "model__" + key: value
                for key, value in self.param_distributions[model_name].items()
            }

            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_distributions,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=make_scorer(f1_score, average="weighted"),
                random_state=self.random_state,
                n_jobs=-1,
            )
            random_search.fit(X, y)

            best_model = random_search.best_estimator_
            y_pred = best_model.predict(X)

            mlflow.log_params(random_search.best_params_)
            mlflow.log_metric("f1_score", random_search.best_score_)
            mlflow.log_metric(
                "precision_score", precision_score(y, y_pred, average="weighted")
            )
            mlflow.log_metric(
                "recall_score", recall_score(y, y_pred, average="weighted")
            )
            mlflow.log_metric("accuracy_score", accuracy_score(y, y_pred))
            mlflow.sklearn.log_model(best_model, "best_model")
            mlflow.sklearn.log_model(best_model.named_steps["model"], model_name)
        return best_model

    def predict(self, X):
        predictions = [model.predict(X) for model in self.best_models_]
        return predictions
