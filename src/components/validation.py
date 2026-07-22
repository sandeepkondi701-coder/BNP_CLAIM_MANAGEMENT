import os
import sys

from dataclasses import dataclass

import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml


@dataclass
class DataValidationConfig:

    validation_status_file_path=str(
        os.path.join(
            "artifacts",
            "validation_status.txt"
        )
    )


class DataValidation:

    def __init__(
            self,
            train_path,
            test_path):

        self.train_path=train_path
        self.test_path=test_path

        self.config=(
            DataValidationConfig()
        )

    def initiate_data_validation(
            self):

        try:

            validation_status=True

            train_df=pd.read_csv(
                self.train_path
            )

            test_df=pd.read_csv(
                self.test_path
            )

            schema=read_yaml(
                "config/schema.yaml"
            )

            print(schema)

            print(type(schema))

            print(schema.keys())

            all_columns=schema[
                "columns"
            ]

            for column in train_df.columns:

                if column not in all_columns.keys():

                    print(f"{column} is missing in schema.yaml")

                    validation_status=False

            with open(
                    self.config.
                    validation_status_file_path,
                    "w"
            ) as f:

                f.write(
                    f"Validation Status:"
                    f"{validation_status}"
                )

            return validation_status

        except Exception as e:

            raise CustomException(
                e,
                sys
            )
        
    def validate_missing_values(self,
                                df):
        
        missing_values = df.isnull().sum()

        print(missing_values)

        return missing_values
    

    def validate_null_percentage(self,df):

        null_percent = (
            df.isnull().mean()
        )*100

        return null_percent
    

    def validate_duplicates(self, df):

        duplicates = (df.duplicated().sum())

        return duplicates
    

    def validate_dtypes(
            self, df, schema):
        
        for col, dtype in schema["columns"].items():

            actual_dtype=(str(
                df[col].dtype
            ))

            print(col, actual_dtype)
        

    def validate_target(
            self, df):
        
        allowed =[0,1]

        for value in df["Target"].unique():

            if value not in allowed:
                return False
            
        
        return True
    
    def validate_statastics(
            self, df):
        
        stats = df.describe()

        print(stats)

        
    def validate_data_drift(
            self, train_df,
            test_df):
        
        for column in train_df.select_dtypes(
            include = "number"):
            
            train_mean = (
                train_df[column].mean()
            )

            test_mean =(
                test_df[column].mean()
            )

            diff = abs(
                train_mean - test_mean
            )

            print(column, 
                  diff
                  )
        


    