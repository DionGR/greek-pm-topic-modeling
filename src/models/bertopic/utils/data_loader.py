import pandas as pd
import pickle
import itertools
from nltk import sent_tokenize
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, speeches_file, statements_file):
        self.speeches_file = speeches_file
        self.statements_file = statements_file
        self.docs = None
        self.sentences = None
        self.train_docs = None
        self.test_docs = None
        self.val_docs = None
        self.train_sentences = None
        self.test_sentences = None
        self.val_sentences = None

    def load_data(self):
        speeches_df = pd.read_csv(self.speeches_file)
        statements_df = pd.read_csv(self.statements_file)
        all_data = pd.concat([speeches_df, statements_df], ignore_index=True)
        self.docs = all_data['text'].dropna().tolist()

    @staticmethod
    def tokenize_sentences(docs):
        sentences = pd.Series(docs).apply(lambda doc: sent_tokenize(doc))
        return list(itertools.chain(*sentences))

    def load_embeddings(self, filename):
        with open(filename, "rb") as fin:
            return pickle.load(fin)

    def get_train_data(self):
        return self.train_docs, self.train_sentences

    def get_test_data(self):
        return self.test_docs, self.test_sentences

    def get_val_data(self):
        return self.val_docs, self.val_sentences

    def split_data(self):
        self.train_docs, self.test_docs = train_test_split(self.docs, test_size=0.15, random_state=1)
        self.train_docs, self.val_docs = train_test_split(self.train_docs, test_size=0.1, random_state=1)

        self.train_sentences = self.tokenize_sentences(self.train_docs)
        self.test_sentences = self.tokenize_sentences(self.test_docs)
        self.val_sentences = self.tokenize_sentences(self.val_docs)

    def process(self):
        self.load_data()
        self.split_data()
