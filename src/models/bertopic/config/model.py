from spacy.lang.el.stop_words import STOP_WORDS as el_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop

STOPWORDS = list(set(el_stop).union(set(en_stop)))

TOP_K = 5
NUM_TOPICS = 30
