from spacy.lang.el.stop_words import STOP_WORDS as el_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop

from models.bertopic.config.embeddings import st_models

from octis.evaluation_metrics.diversity_metrics import TopicDiversity, KLDivergence
from octis.evaluation_metrics.similarity_metrics import RBO, PairwiseJaccardSimilarity

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

STOPWORDS = list(set(el_stop).union(set(en_stop)))

TOP_K = 5
TOP_N_WORDS = 10
NUM_TOPICS = 30
N_GRAMS = (1, 2)
EMBEDDING_MODEL = st_models["gr_media"]

""" UMAP Model parameters """
N_NEIGHBORS_DOCS = 5
N_NEIGHBORS_SENTENCES = 10

""" HDBSCAN Model parameters """
MIN_CLUSTER_SIZE_DOCS = 10
MIN_CLUSTER_SIZE_SENTENCES = 5
MIN_SAMPLES_DOCS = 5


metrics = {
    'coherence_c_npmi': None,
    'coherence_c_v': None,
    'coherence_u_mass': None,
    'coherence_c_uci': None,
    'diversity_topic': TopicDiversity(topk=TOP_K),
    'similarity_rbo': RBO(topk=TOP_K),
    'similarity_pjs': PairwiseJaccardSimilarity(),
}



umap_configs_dc = [
    {"n_neighbors": 2},
    {"n_neighbors": 3},
    {"n_neighbors": 5},
    {"n_components": 2},
    {"n_components": 3},
    {"n_components": 5},
    {"n_neighbors": 2, "n_components": 2},
    {"n_neighbors": 3, "n_components": 3},
    {"n_neighbors": 5, "n_components": 5},
    {"n_neighbors": 2, "n_components": 2, "min_dist": 0.001},
    {"n_neighbors": 3, "n_components": 3, "min_dist": 0.01},
    {"n_neighbors": 5, "n_components": 5, "min_dist": 0.1}
]

hdbscan_configs_dc = [
    {"min_cluster_size": 5},
    {"min_cluster_size": 10},
    {"min_cluster_size": 15},
    {"min_cluster_size": 10, "min_samples": 5},
    {"min_cluster_size": 10, "min_samples": 15}
]

umap_configs_sc = [
    {"n_neighbors": 3},
    {"n_neighbors": 5},
    {"n_neighbors": 10},
    {"n_components": 3},
    {"n_components": 5},
    {"n_components": 10}
]

hdbscan_configs_sc = [
    {"min_cluster_size": 5},
    {"min_cluster_size": 10},
    {"min_cluster_size": 15},
    {"min_cluster_size": 30}
]

""" Different model configurations for BERTopic """

dim_models = {
    "PCA_2": PCA(n_components=2),
    "PCA_5": PCA(n_components=5),
    "TSNE_2": TSNE(n_components=2),
    "TSNE_3": TSNE(n_components=3),
}

cluster_models = {
    "KMeans": KMeans(n_clusters=NUM_TOPICS),
    "GaussianMixture": GaussianMixture(n_components=NUM_TOPICS)
}
