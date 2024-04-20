from octis.models.model import AbstractModel
from octis.evaluation_metrics.metrics import AbstractMetric
from octis.evaluation_metrics.coherence_metrics import Coherence

import pandas as pd

from typing import Dict, List


class BERTopicModelEvaluator:

    def __init__(self, 
                 models: Dict[str, AbstractModel], 
                 metrics: Dict[str, AbstractMetric],
                 dataset: List[str],
                 topics: int = 10, 
                 top_k: int = 7,
                 granularity: str = 'docs',
                 mode: str = 'optimize'
                 ):

        self.models = models
        self.metrics = metrics
        self.dataset = dataset
        self.trained = False
        self.topics = topics
        self.top_k = top_k
        self.model_outputs = {}
        self.model_topics = {}
        self.granularity = granularity
        self.mode = mode
        
        self.evaluated = {}
        for model_type, _ in self.models.items():
            self.evaluated[model_type] = False

        evaluation_cols = []
        for metric_type, _ in self.metrics.items():
            evaluation_cols.append(metric_type)

        self.evaluation_df = pd.DataFrame(columns=["model"] + evaluation_cols)

        topic_cols = []
        for topic in range(self.top_k):
            topic_cols.append("topic_" + str(topic))

        self.topics_df = pd.DataFrame(columns=["model"] + topic_cols)


    def evaluate(self):
        for model_type, model in self.models.items():
            
            print("Training model: ", model_type)
            topics, _ = model.fit_transform(self.dataset)
            self.model_outputs[model_type] = model
            self.model_topics[model_type] = topics
            print("Model training complete.")

            
            print("Evaluating model: ", model_type)
            model_output_list = self.generate_topics_list(self.model_topics[model_type], model)
            model_output = {'topics': model_output_list}
            model_metric_data = {'model': [model_type]}

            if model_output['topics'] is None or len(model_output['topics']) == 1:
                print(f"Skipping evaluation for model {model_type} as no topics were generated.")
                continue

            tokens = self.get_tokens(model, self.dataset, self.model_topics[model_type])
            
            if tokens is None or len(tokens) == 1 or len(tokens) == 0:
                print(f"Skipping evaluation for model {model_type} as no tokens were generated.")
                continue

            for metric_type in self.metrics.keys():
                if metric_type.startswith('coherence_'):
                    measure = metric_type[len('coherence_'):]
                    self.metrics[metric_type] = Coherence(topk=self.top_k, measure=measure, texts=tokens)

            for metric_type, metric in self.metrics.items():
                print(f"Evaluating metric {metric_type} for model {model_type}")
                score = metric.score(model_output)
                model_metric_data[metric_type] = [score]

            metric_df = pd.DataFrame(model_metric_data)
            topics_df = self.get_model_topics_df(model_type)
            
            total_df = pd.concat([metric_df, topics_df], axis=1)
            
            self.evaluated[model_type] = True
            self.evaluation_df = pd.concat([self.evaluation_df, metric_df], ignore_index=True)
            print(f"Model {model_type} evaluated, generated {len(model_output_list)} topics.")
            
            if self.mode == 'optimize':
                total_df.to_csv(f"data/optimization/{self.granularity}_gran/individual/{model_type}.csv")
            else:
                total_df.to_csv(f"models/bertopic/data/evaluation/final_evaluation_results.csv")

        # self.export_metrics(f"models/bertopic/data/evaluation/{self.granularity}_gran/evaluation_results.csv")
        # self.export_topics(f"models/bertopic/data/evaluation/{self.granularity}_gran/topics_results.csv")

        if self.mode == 'optimize':
            return self.evaluation_df, self.topics_df
        else:
            return self.evaluation_df, self.models
    
    def export_df(self, df, path):
        df.to_csv(path)

    def export_metrics(self, path):
        self.evaluation_df.to_csv(path)
        
    def get_model_topics_df(self, model_type):
        topic_data = {"model": [model_type]}
        topics = self.model_topics[model_type]
        topic_words = self.generate_topics_list(topics, self.model_outputs[model_type])
        if topic_words is None:
            return None
        for i, topic in enumerate(topic_words):
            topic_data["topic_" + str(i)] = [topic]
            
        topics_df = pd.DataFrame(topic_data)
        
        return topics_df

    def export_topics(self, path):
        for model_type, model in self.models.items():
            print("Exporting topics for model: ", model_type)

            if not self.evaluated[model_type]:
                print(f"Skipping topic export for model {model_type} as it was not evaluated.")
                continue

            topics_df = self.get_model_topics_df(model_type)
                
            self.topics_df = pd.concat([self.topics_df, topics_df], ignore_index=True)
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
