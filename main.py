from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import (DataIngestionConfig, 
                                                  DataValidationConfig, 
                                                  DataTransformationConfig, 
                                                  ModelTrainerConfig
                                                )
from networksecurity.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == '__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()

        ## Data Ingestion
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        dataingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact = dataingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")
        print(dataingestionartifact,"\n")

        ## Data Validation
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact = data_validation.initiate_data_ingestion()
        logging.info("Data validation Completed")
        print(data_validation_artifact ,"\n")

        # Data Transformation
        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Initiate the Data transformation")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation Completed")
        print(data_transformation_artifact ,"\n")

        # Model Training
        logging.info("Model Training Started...")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                     data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        logging.info("Model Training Artifact Created")
        print(model_trainer_artifact ,"\n")

        
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
