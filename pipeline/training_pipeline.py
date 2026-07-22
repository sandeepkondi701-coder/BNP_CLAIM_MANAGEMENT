from src.components.ingestion import DataIngestion
from src.components.validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    # Data Ingestion
    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    # Data Validation
    validation = DataValidation(
        train_path,
        test_path
    )

    validation.initiate_data_validation()

    # Data Transformation
    transformation = DataTransformation()

    X_train, X_test, y_train, y_test = (
        transformation.
        initiate_data_transformation(
            train_path,
            test_path
        )
    )

    # Model Training
    trainer = ModelTrainer()

    score = (
        trainer.initiate_model_trainer(
            X_train,
            y_train,
            X_test,
            y_test
        )
    )

    print(
        f"\nBest F1 Score: {score}"
    )
