from octis.models.model import AbstractModel
from octis.evaluation_metrics.metrics import AbstractMetric
from octis.evaluation_metrics.coherence_metrics import Coherence

import pandas as pd

from typing import Dict, List


class BERTopicModelEvaluator:

    def __init__(self, 
                 models: Dict[str, AbstractModel], 
                 metrics: Dict[str, AbstractMetric],
                 train_dataset: List[str] = [],
                 test_dataset: List[str] = [],
                 val_dataset: List[str] = [],
                 topics: int = 10, 
                 topk: int = 5,
                 eval_type: str = 'val',
                 granularity: str = 'docs'
                 ):

        self.models = models
        self.metrics = metrics
        self.datasets = {'train': train_dataset, 'test': test_dataset, 'val': val_dataset}
        self.trained = False
        self.topics = topics
        self.topk = topk
        self.model_outputs = {}
        self.model_topics = {}
        self.eval_type = eval_type
        self.granularity = granularity
        
        self.evaluated = {}
        for model_type, _ in self.models.items():
            self.evaluated[model_type] = False

        evaluation_cols = []
        for metric_type, _ in self.metrics.items():
            evaluation_cols.append(metric_type)

        self.evaluation_df = pd.DataFrame(columns=["model"] + evaluation_cols)

        topic_cols = []
        for topic in range(self.topk):
            topic_cols.append("topic_" + str(topic))

        self.topics_df = pd.DataFrame(columns=["model"] + topic_cols)

    def train(self):
        for model_type, model in self.models.items():
            print("Training model: ", model_type)
            topics, _ = model.fit_transform(self.datasets['train'])
            print("Model training complete.")
            
            self.model_topics[model_type] = topics

        self.trained = True

    def evaluate(self):
        if not self.trained:
            self.train()

        for model_type, model in self.models.items():
            print("Evaluating model: ", model_type)
            
            model_output = model.transform(self.datasets[self.eval_type])
            
            model_metric_data = {'model': [model_type], 'dataset': [self.datasets[self.eval_type]]}

            if model_output['topics'] is None:
                print(f"Skipping evaluation for model {model_type} as no topics were generated.")
                continue

            tokens = self.get_tokens(model, self.datasets[self.eval_type], self.model_topics[model_type])

            for metric_type in self.metrics.keys():
                if metric_type.startswith('coherence_'):
                    measure = metric_type[len('coherence_'):]
                    self.metrics[metric_type] = Coherence(topk=self.topk, measure=measure, texts=tokens)

            for metric_type, metric in self.metrics.items():
                print(f"Evaluating metric {metric_type} for model {model_type}")
                score = metric.score(model_output)
                model_metric_data[metric_type] = [score]

            metric_df = pd.DataFrame(model_metric_data)
            self.evaluated[model_type] = True
            self.evaluation_df = pd.concat([self.evaluation_df, metric_df], ignore_index=True)
            print(f"Model {model_type} evaluated")
            
            metric_df.to_csv(f"models/bertopic/data/evaluation/{self.granularity}_gran/{model_type}.csv")

        self.export_metrics(f"models/bertopic/data/evaluation/{self.granularity}_gran/evaluation_results.csv")
        self.export_topics(f"models/bertopic/data/evaluation/{self.granularity}_gran/topics_results.csv")

        return self.evaluation_df
    
    def export_df(self, df, path):
        df.to_csv(path)

    def export_metrics(self, path):
        self.evaluation_df.to_csv(path)

    def export_topics(self, path):
        for model_type, model in self.models.items():
            print("Exporting topics for model: ", model_type)

            if not self.evaluated[model_type]:
                print(f"Skipping topic export for model {model_type} as it was not evaluated.")
                continue

            topic_data = {"model": [model_type]}
            topics = self.model_topics[model_type]
            topic_words = self.generate_topics_list(topics, model)
            for i, topic in enumerate(topic_words):
                topic_data["topic_" + str(i)] = [topic]  # Wrap topic in a list
            self.topics_df = pd.concat([self.topics_df, pd.DataFrame(topic_data)], ignore_index=True)
        self.topics_df.to_csv(path)

    @staticmethod
    def generate_topics_list(topics, model):
        num_unique_topics = len(set(topics))
        if num_unique_topics == 1:
            print(f"Warning: Only one unique topic was generated by the model.")
            print("No metrics can be calculated.")
            topic_words = None
        else:
            topic_words = [[words for words, _ in model.get_topic(topic)] for topic in range(num_unique_topics - 1)]
        return topic_words

    @staticmethod
    def get_tokens(model, dataset, topics):
        documents = pd.DataFrame({"Document": dataset,
                                  "ID": range(len(dataset)),
                                  "Topic": topics})

        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

        vectorizer = model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        tokens = [analyzer(doc) for doc in cleaned_docs]
        return tokens
