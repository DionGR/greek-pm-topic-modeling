from spacy.lang.el.stop_words import STOP_WORDS as el_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop

from models.bertopic.config.embeddings import st_models

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.similarity_metrics import RBO, PairwiseJaccardSimilarity


TOP_K = 5
NUM_TOPICS = 30
STOPWORDS = list(set(el_stop).union(set(en_stop)))

metrics = {
    'coherence_c_npmi': None,
    'coherence_c_v': None,
    'coherence_u_mass': None,
    'coherence_c_uci': None,
    'diversity_topic': TopicDiversity(topk=TOP_K),
    'similarity_rbo': RBO(topk=TOP_K),
    'similarity_pjs': PairwiseJaccardSimilarity(),
}

""" SentenceTransformer parameters """
EMBEDDING_MODEL = st_models["gr_media"]

""" CountVectorizer parameters """
vectorizer_params = {
    "ngram_range": (1, 2),
    "stop_words": STOPWORDS,
    "max_df": 0.85,
    "min_df": 0.01,
}

""" c-TF-IDF parameters """
c_tfidf_params = {
    "reduce_frequent_words": True
}
