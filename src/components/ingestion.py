import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )

    raw_data_path: str = os.path.join(
        "artifacts",
        "raw.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered Data Ingestion Method")

        try:

            # Read Dataset
            df = pd.read_csv(
                "data/raw/train.csv"
            )

            logging.info(
                "Dataset Loaded Successfully"
            )

            # Create artifacts folder if it doesn't exist
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            # Save raw dataset
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Raw Data Saved Successfully"
            )

            # Train-Test Split
            train_set, test_set = train_test_split(
                df,
                test_size=0.20,
                random_state=42
            )

            logging.info(
                "Train-Test Split Completed"
            )

            # Save train data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Save test data
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Train and Test Files Saved Successfully"
            )

            logging.info(
                "Data Ingestion Completed"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)