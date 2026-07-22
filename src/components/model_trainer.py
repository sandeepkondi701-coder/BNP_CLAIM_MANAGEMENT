import os
import sys
import dill

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV
)

from src.logger import logging
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = (
            ModelTrainerConfig()
        )

    def evaluate_models(
            self,
            X_train,
            y_train,
            X_test,
            y_test):

        try:

            models = {

                "Logistic Regression":
                    LogisticRegression(),

                "KNN":
                    KNeighborsClassifier(),

                "Decision Tree":
                    DecisionTreeClassifier(),

                "Random Forest":
                    RandomForestClassifier(),

                "Gradient Boosting":
                    GradientBoostingClassifier()

            }

            report = {}

            for name, model in models.items():

                logging.info(
                    f"Training {name}"
                )

                # Train Model
                model.fit(
                    X_train,
                    y_train
                )

                # Predictions
                y_pred = model.predict(
                    X_test
                )

                # Metrics
                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                precision = precision_score(
                    y_test,
                    y_pred
                )

                recall = recall_score(
                    y_test,
                    y_pred
                )

                f1 = f1_score(
                    y_test,
                    y_pred
                )

                roc_auc = roc_auc_score(
                    y_test,
                    y_pred
                )

                # Cross Validation
                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="f1"
                )

                report[name] = {

                    "Accuracy":
                        accuracy,

                    "Precision":
                        precision,

                    "Recall":
                        recall,

                    "F1":
                        f1,

                    "ROC_AUC":
                        roc_auc,

                    "CV Score":
                        cv_scores.mean()
                }

            return report

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def initiate_model_trainer(
            self,
            X_train,
            y_train,
            X_test,
            y_test
    ):

        try:

            model_report = (
                self.evaluate_models(
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )
            )

            print("\nModel Report\n")

            for key, value in (
                    model_report.items()
            ):

                print(
                    key,
                    value
                )

            # Select Best Model
            best_model_name = max(
                model_report,
                key=lambda x:
                model_report[x]["F1"]
            )

            print(
                f"\nBest Model:"
                f" {best_model_name}"
            )

            # Hyperparameter Tuning
            if best_model_name == (
                    "Random Forest"
            ):

                params = {

                    "n_estimators":
                        [100, 200, 300],

                    "max_depth":
                        [5, 10, 20]
                }

                grid = GridSearchCV(

                    RandomForestClassifier(),

                    param_grid=params,

                    cv=5,

                    scoring="f1"

                )

                grid.fit(
                    X_train,
                    y_train
                )

                best_model = (
                    grid.best_estimator_
                )

                print(
                    "\nBest Parameters:"
                )

                print(
                    grid.best_params_
                )

                print(
                    "\nBest CV Score:"
                )

                print(
                    grid.best_score_
                )

            else:

                models = {

                    "Logistic Regression":
                        LogisticRegression(),

                    "KNN":
                        KNeighborsClassifier(),

                    "Decision Tree":
                        DecisionTreeClassifier(),

                    "Random Forest":
                        RandomForestClassifier(),

                    "Gradient Boosting":
                        GradientBoostingClassifier()
                }

                best_model = (
                    models[
                        best_model_name
                    ]
                )

                best_model.fit(
                    X_train,
                    y_train
                )

            # Save Model
            with open(
                    self.
                    model_trainer_config.
                    trained_model_file_path,
                    "wb"
            ) as file_obj:

                dill.dump(
                    best_model,
                    file_obj
                )

            logging.info(
                "Model Saved Successfully"
            )

            print(
                "\nModel Saved:"
                " artifacts/model.pkl"
            )

            return (
                model_report[
                    best_model_name
                ]["F1"]
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )
