from typing import List, Dict
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import itertools


class STEmbeddingTrainer:
    def __init__(self, st_model_types: Dict[str, str], dataset, save_path: str):
        self.st_model_types = st_model_types
        self.dataset = dataset
        self.save_path = save_path

    def train_all(self):
        for model_name, model_link in self.st_model_types.items():
            sentence_transformer = SentenceTransformer(model_link)
            embeddings = sentence_transformer.encode(sentences=self.dataset, 
                                                     show_progress_bar=True,
                                                     batch_size=64
                                                     )
                                                     
            sentence_transformer.save(self.save_path + f"/{model_name}", model_name, safe_serialization=True)
            # self.save_embeddings(embeddings, f"{self.save_path}/{model_name}.pkl")

        

