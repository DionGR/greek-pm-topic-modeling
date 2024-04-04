from octis.models.model import AbstractModel
from octis.evaluation_metrics.metrics import AbstractMetric
from octis.dataset.dataset import Dataset

import pandas as pd

from typing import Dict

class OCTISModelEvaluator: 
    
    def __init__(self, models: Dict[str, AbstractModel], dataset: Dataset, metrics: Dict[str, AbstractMetric]):
        self.models = models
        self.dataset = dataset
        self.metrics = metrics
        self.model_outputs = {}
        self.trained = False

        evaluation_cols = []
        for metric_type, _ in self.metrics.items():
            evaluation_cols.append(metric_type)
            
        self.evaluation_df = pd.DataFrame(columns=["model"] + evaluation_cols)
        
    def train(self):
        for model_type, model in self.models.items():
            self.model_outputs[model_type] = model.train_model(self.dataset)
            
        self.trained = True
        
    def evaluate(self):
        if not self.trained:
            self.train()
                    
        for model_type, model_output in self.model_outputs.items():
            model_metric_data = {"model": [model_type]}
            for metric_type, metric in self.metrics.items():
                metric_results = metric.score(model_output)
                model_metric_data[metric_type] = [metric_results]
                
            self.evaluation_df = pd.concat([self.evaluation_df, pd.DataFrame(model_metric_data)], ignore_index=True)
                
        self.export_results("models/octis/data/evaluation/evaluation_results.csv")
                
        return self.evaluation_df
    
    def export_results(self, path):
        self.evaluation_df.to_csv(path)
