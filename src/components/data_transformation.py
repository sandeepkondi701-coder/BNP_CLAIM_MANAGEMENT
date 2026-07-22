import os
import sys

import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from src.exception import CustomException
from src.logger import logging

import dill


@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):

        self.data_transformation_config = (
            DataTransformationConfig()
        )

    def get_data_transformer_object(self):

        try:

            numerical_columns = [
                "customer_age",
                "annual_premium"
            ]

            categorical_columns = [
                "claim_type"
            ]

            logging.info(
                "Creating Numerical Pipeline"
            )

            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            logging.info(
                "Creating Categorical Pipeline"
            )

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "encoder",
                        OneHotEncoder()
                    )
                ]
            )

            logging.info(
                "Creating Column Transformer"
            )

            preprocessor = ColumnTransformer(
                [
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),

                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            return preprocessor

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def initiate_data_transformation(
            self,
            train_path,
            test_path
    ):

        try:

            train_df = pd.read_csv(
                train_path
            )

            test_df = pd.read_csv(
                test_path
            )

            logging.info(
                "Train and Test Data Loaded"
            )

            target_column = "target_processed"

            X_train = train_df.drop(
                columns=[target_column],
                axis=1
            )

            y_train = train_df[
                target_column
            ]

            X_test = test_df.drop(
                columns=[target_column],
                axis=1
            )

            y_test = test_df[
                target_column
            ]

            preprocessing_obj = (
                self.
                get_data_transformer_object()
            )

            logging.info(
                "Applying fit_transform on train data"
            )

            X_train_arr = (
                preprocessing_obj.
                fit_transform(
                    X_train
                )
            )

            logging.info(
                "Applying transform on test data"
            )

            X_test_arr = (
                preprocessing_obj.
                transform(
                    X_test
                )
            )

            logging.info(
                "Saving Preprocessor Object"
            )

            with open(
                    self.
                    data_transformation_config.
                    preprocessor_obj_file_path,
                    "wb"
            ) as file_obj:

                dill.dump(
                    preprocessing_obj,
                    file_obj
                )

            logging.info(
                "Data Transformation Completed"
            )

            return (

                X_train_arr,
                X_test_arr,
                y_train,
                y_test

            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )