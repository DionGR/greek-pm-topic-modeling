#!/usr/bin/env python
# coding: utf-8

# # Finding the Best Hyperparameters for BERTopic

# In this notebook we will try to find the best hyperparameters for our BERTopic model, by trying different configurations of UMAP and HDBSCAN models. Then we will evaluate each model based on both standard evaluation metrics and manual inspection of the topics created. 

# ## 

# ## Imports & Setup

# In[ ]:


print("starting")
import nltk
nltk.download('punkt')
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from sentence_transformers import SentenceTransformer

from utils.data_loader import DataLoader
from utils.bertopic_evaluator import BERTopicModelEvaluator
from config.model import NUM_TOPICS, TOP_K, EMBEDDING_MODEL, vectorizer_params, c_tfidf_params, metrics
from config.optimization import all_config_combinations, algos_dict

from bertopic.vectorizers import ClassTfidfTransformer


# In[ ]:


print("starting2")
loader = DataLoader('../../data/data_speeches.csv', '../../data/data_statements.csv', split_data=False)
loader.process()

train_docs, train_sentences = loader.get_train_data()


# For this step, we only need the training and validation data.

# ## Constant Model Initialization

# We will use the same vectorizer, c-TF-IDF model and sentence transformer model for all our experiments. 
# 
# We're interesting in optimizing on the Dimensionality Reduction and Clustering models, so we will keep the rest of the pipeline constant.

# In[ ]:


vectorizer_model = CountVectorizer(**vectorizer_params)
ctfidf_model = ClassTfidfTransformer(**c_tfidf_params)
st_model = SentenceTransformer(EMBEDDING_MODEL)


# Evaluating different UMAP and HDBSCAN configurations

# ## Hyperparameter Tuning

# Let's see how many different configurations we can try:

# In[ ]:


search_space = all_config_combinations()
len(search_space)


# ### Document Level

# In[ ]:


# granularity = 'docs'


# In[ ]:


# models = {}

print("Loading models...")

# for config in search_space:
#     dim_reduction_model = algos_dict[config['dim_reduction_model']](**config['dim_reduction_params'])
#     clustering_model = algos_dict[config['clustering_model']](**config['clustering_params'])
        
#     model_name = f"model_{config['dim_reduction_model']}_{config['dim_reduction_params']}_{config['clustering_model']}_{config['clustering_params']}"


#     model = BERTopic(
#         umap_model=dim_reduction_model,
#         hdbscan_model=clustering_model,
#         embedding_model=EMBEDDING_MODEL,
#         vectorizer_model=vectorizer_model,
#         ctfidf_model=ctfidf_model,
#         nr_topics=NUM_TOPICS,
#         top_n_words=TOP_K,
#     )
    
#     models[model_name] = model


# In[ ]:


# evaluator = BERTopicModelEvaluator(
#                                    models=models, 
#                                    metrics=metrics, 
#                                    dataset=train_docs,
#                                    topics=NUM_TOPICS,
#                                    top_k=TOP_K,
#                                    granularity=granularity
#                                    )                    


# In[ ]:


print("Evaluating models...")
# evaluator.evaluate()


# ### Sentence Level

# In[ ]:


granularity = 'sents'


# In[ ]:


models = {}

for config in search_space:
    dim_reduction_model = algos_dict[config['dim_reduction_model']](**config['dim_reduction_params'])
    clustering_model = algos_dict[config['clustering_model']](**config['clustering_params'])
        
    model_name = f"model_{config['dim_reduction_model']}_{config['dim_reduction_params']}_{config['clustering_model']}_{config['clustering_params']}"


    model = BERTopic(
        umap_model=dim_reduction_model,
        hdbscan_model=clustering_model,
        embedding_model=EMBEDDING_MODEL,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        nr_topics=NUM_TOPICS,
        top_n_words=TOP_K,
    )
    
    models[model_name] = model


# In[ ]:


evaluator = BERTopicModelEvaluator(
                                   models=models, 
                                   metrics=metrics, 
                                   dataset=train_sentences,
                                   topics=NUM_TOPICS,
                                   top_k=TOP_K,
                                   granularity=granularity
                                   )                    


# In[ ]:


evaluator.evaluate()

