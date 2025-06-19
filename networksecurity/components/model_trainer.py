import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
import mlflow

import dagshub
dagshub.init(repo_owner='forcoding247', repo_name='Network-Secutiry', mlflow=True)


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, classsification_metrics):
        with mlflow.start_run():
            f1_score = classsification_metrics.f1_score
            recall_score = classsification_metrics.recall_score
            precision_score = classsification_metrics.precision_score

            ## Log the Metrics
            mlflow.log_metric("f1_score", f1_score, )
            mlflow.log_metric("recall_score", recall_score, )
            mlflow.log_metric("Precision_score",precision_score, )

            ## Log the parameters
            if hasattr(best_model, "get_params"):
                params = best_model.get_params()
                mlflow.log_params(params)


        
    def train_model(self, x_train, y_train, x_test, y_test):
        models = {
            "Random Forest" : RandomForestClassifier(n_jobs=-1, verbose=1),
            "Gradient Boost" : GradientBoostingClassifier(verbose=1),
            "Logistic Regression" : LogisticRegression(verbose=1),
            "Adaboost" : AdaBoostClassifier(),
            "Decision Tree" : DecisionTreeClassifier(),
        }

        params = {
            "Decision Tree" : {
                'criterion' : ['gini','entropy', 'log_loss'],
                # 'splitter' : ["best", "random"],
                # "max_features":['sqrt', 'log2'],
            },
            "Random Forest" : {
                'criterion' : ['gini','entropy', 'log_loss'],
                # "max_features":['sqrt', 'log2', None],
                "n_estimators" : [8, 16, 32, 64, 128, 256],
            },
            "Gradient Boost" : {
                # "loss" : ["log_loss", "exponential"],
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                # "criterion": ["friedman_mse", "squared_error"],
                # "max_features":['sqrt', 'log2', auto],
                "n_estimators" : [8, 16, 32, 64, 128, 256],
            },
            "Logistic Regression":{
            },
            "Adaboost":{
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "n_estimators" : [8, 16, 32, 64, 128, 256],
            }
        }

        model_report: dict = evaluate_models(X_train=x_train, y_train= y_train,
                                             X_test= x_test, y_test= y_test, models= models,
                                             params= params)
        
        ## To get the best model score 
        best_model_score = max(sorted(model_report.values()))

        ## to get the best model name
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        y_train_pred = best_model.predict(x_train)
        classification_train_metric = get_classification_report(y_true=y_train, y_pred= y_train_pred)

        ##track the experiments with MLFlow

        self.track_mlflow(best_model, classification_train_metric)


        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_report(y_true=y_test, y_pred= y_test_pred)
        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_path_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_path_dir, exist_ok=True)

        Network_Model = NetworkModel(preprocessor=preprocessor, model= best_model)

        save_object(self.model_trainer_config.trained_model_file_path, Network_Model)

        ## Model Trainer Artifact
        model_trainer_artifact =ModelTrainerArtifact(trained_model_file_path = self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact = classification_train_metric,
                             test_metric_artifact = classification_test_metric,
                             )
        
        logging.info(f"Model Trainer Artifact {model_trainer_artifact}")

        return model_trainer_artifact


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            ## Loading the training and testing array
            train_arr = load_numpy_array(train_file_path)
            test_arr = load_numpy_array(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:,-1],
                test_arr[:, :-1],
                test_arr[:,-1]
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

