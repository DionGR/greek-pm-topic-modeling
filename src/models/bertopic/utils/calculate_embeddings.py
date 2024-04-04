import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import itertools
import pickle


class EmbeddingsGenerator:
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

    @staticmethod
    def get_sentences(text):
        return sent_tokenize(text)

    def tokenize_sentences(self):
        sentences = pd.Series(self.docs).apply(self.get_sentences)
        self.sentences = list(itertools.chain(*sentences))

    def generate_embeddings(self):
        st_greek = SentenceTransformer('lighteternal/stsb-xlm-r-greek-transfer')
        st_greek_media = SentenceTransformer('dimitriz/st-greek-media-bert-base-uncased')

        embeddings_gr = st_greek.encode(self.docs, show_progress_bar=True)
        embeddings_gr_media = st_greek_media.encode(self.docs, show_progress_bar=True)
        embeddings_gr_sentences = st_greek.encode(self.sentences, show_progress_bar=True)
        embeddings_gr_media_sentences = st_greek_media.encode(self.sentences, show_progress_bar=True)

        return embeddings_gr, embeddings_gr_media, embeddings_gr_sentences, embeddings_gr_media_sentences

    @staticmethod
    def save_embeddings(embeddings, filename):
        with open(filename, "wb") as fout:
            pickle.dump(embeddings, fout)

    def process(self):
        self.load_data()
        self.tokenize_sentences()
        embeddings = self.generate_embeddings()
        self.save_embeddings(embeddings[0], "embeddings_gr.pkl")
        self.save_embeddings(embeddings[1], "embeddings_gr_media.pkl")
        self.save_embeddings(embeddings[2], "embeddings_gr_sentences.pkl")
        self.save_embeddings(embeddings[3], "embeddings_gr_media_sentences.pkl")


if __name__ == "__main__":
    generator = EmbeddingsGenerator('data/data_speeches.csv', 'data/data_statements.csv')
    generator.process()
