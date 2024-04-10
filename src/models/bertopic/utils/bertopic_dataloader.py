import pandas as pd
import pickle
import itertools

from nltk import sent_tokenize


class BERTopicDataloader:
    def __init__(self, speeches_file, statements_file):
        self.speeches_file = speeches_file
        self.statements_file = statements_file
        self.docs = None
        self.sentences = None

    def load_data(self):
        speeches_df = pd.read_csv(self.speeches_file)
        statements_df = pd.read_csv(self.statements_file)
        all_data = pd.concat([speeches_df, statements_df], ignore_index=True)
        self.docs = all_data['text'].dropna().tolist()

    def tokenize_sentences(self):
        sentences = pd.Series(self.docs).apply(lambda doc: sent_tokenize(doc))
        self.sentences = list(itertools.chain(*sentences))

    def process(self):
        self.load_data()
        self.tokenize_sentences()

    def load_embeddings(self, filename):
        with open(filename, "rb") as fin:
            return pickle.load(fin)

    def get_docs(self):
        return self.docs

    def get_sentences(self):
        return self.sentences
