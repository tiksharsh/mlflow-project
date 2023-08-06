import pandas as pd
import os
import json
from mlproject import logger
from sklearn.linear_model import ElasticNet
import joblib
from mlproject.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        # Report training set score
        train_score = lr.score(train_x, train_y) * 100
        test_score = lr.score(test_x, test_y) * 100
        with open(self.config.score_file_name, "w") as fd:
            json.dump(
                {
                    "Train_score": train_score.tolist(),
                    "Test_score": test_score.tolist()
                   
                }, fd, indent=4)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))

