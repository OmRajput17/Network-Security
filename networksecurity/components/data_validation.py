from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


## Configration file from data ingestion config

from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH

import os
import sys
import pandas as pd
from scipy.stats import ks_2samp
from networksecurity.utils.main_utils.utils import read_yaml


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.SCHEMA_FILE_PATH = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_data_ingestion(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            ## Read the data from train and test 
        except Exception as e:
            raise NetworkSecurityException(e, sys)