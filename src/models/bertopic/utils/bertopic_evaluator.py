from octis.models.model import AbstractModel
from octis.evaluation_metrics.metrics import AbstractMetric
from octis.dataset.dataset import Dataset

from models.octis.utils.model_evaluator import OCTISModelEvaluator

from bertopic import BERTopic

import pandas as pd

from typing import Dict


class BERTopicModelEvaluator:

    def __init__(self, models: Dict[str, AbstractModel], metrics: Dict[str, AbstractMetric],
                 datasets: Dict[str, Dataset], topics: int = 10):

        self.models = models
        self.metrics = metrics
        self.datasets = datasets
        self.model_outputs = {}
        self.trained = False
        self.topics = topics

        evaluation_cols = []
        for metric_type, _ in self.metrics.items():
            evaluation_cols.append(metric_type)

        self.evaluation_df = pd.DataFrame(columns=["model"] + evaluation_cols)

    def train(self):
        for model_type, model in self.models.items():
            self.model_outputs[model_type] = model

        self.trained = True

    def evaluate(self):
        if not self.trained:
            self.train()

        for model_type, model_output in self.model_outputs.items():
            model_output = self.convert_bertopipc_output(model_output)
            dataset = model_type.split('_')[-1]
            model_metric_data = {'model': [model_type], 'dataset': [dataset]}

            for metric_type, metric in self.metrics['coherence_metrics'][dataset].items():
                model_metric_data[metric_type] = [metric.score(model_output)]

            for metric_type, metric in self.metrics['other_metrics'].items():
                model_metric_data[metric_type] = [metric.score(model_output)]

            self.evaluation_df = pd.concat([self.evaluation_df, pd.DataFrame(model_metric_data)], ignore_index=True)

        self.export_metrics("models/bertopic/data/evaluation/evaluation_results.csv")

        return self.evaluation_df

    def export_metrics(self, path):
        self.evaluation_df.to_csv(path)


    def convert_bertopipc_output(self, bertopic_model):
        topics_dict = bertopic_model.get_topics()
        topics_list = []
        for topic_id, words in topics_dict.items():
            topic_words = [word for word, _ in topics_dict[topic_id]]
            topics_list.append(topic_words)

        return topics_list
