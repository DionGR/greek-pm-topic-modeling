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
        
        self.evaluation_df = pd.DataFrame(columns=["model", "metric", "value"])
        self.model_outputs = {}

        self.trained = False
        
    def train(self):
        for model_type, model in self.models:
            self.model_outputs[model_type] = model.train(self.dataset)
            
        self.trained = True
        
    def evaluate(self):
        if not self.trained:
            self.train()
                    
        for model_type, model_output in self.model_outputs:
            for metric_type, metric in self.metrics:
                metric_results = metric.score(model_output)
                
                self.evaluation_df = self.evaluation_df.append(pd.DataFrame({"model": model_type, "metric": metric_type, "value": metric_results}))
                
        return self.evaluation_df
