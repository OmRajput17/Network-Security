import os,sys
import yaml
import dill
from pathlib import Path
from networksecurity.logging.logger import logging
import numpy as np
import pickle
from networksecurity.exception.exception import NetworkSecurityException



def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

