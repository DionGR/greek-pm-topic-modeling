from spacy.lang.el.stop_words import STOP_WORDS as el_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop

STOPWORDS = list(set(el_stop).union(set(en_stop)))

TOP_K = 5
NUM_TOPICS = 30
N_GRAMS = (1, 2) 

""" BERTopic Configurations """
 
 
""" BERTopic Sub-Model Configruations"""

vectorizer_params_dc = {}
vectorizer_params_st = {}

