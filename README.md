# Comparative Analysis of Generative and Neural Topic Modeling Techniques Applied to Modern Greek Political Corpora

## Abstract

In this project we compare the performance of generative and neural topic modeling techniques applied to a modern Greek political corpus. We use speeches and statements from the Office of the Prime Minister of Greece to create a corpus of approximately 2000 documents. We optimize and experiment with a range of Topic Modeling techniques, including Latent Semantic Analysis (LSA), Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF), Hierarchical Dirichlet Process (HDP), ProdLDA, and BERTopic. We evaluate the performance of these techniques using a range of metrics, including coherence, perplexity, and topic interpretability. We also explore the impact of preprocessing steps, such as lemmatization, stopword removal, and bigram extraction, on the performance of these techniques. 

## How to run the code
1. Clone the repository
2. Install the required packages from the requirements.txt file
3. Run the code in the following order:
    - [OPTIONAL] Run `load_dataset.ipynb` if you do **not** wish to use the dataset we provide.
        - Additionally, you will need to re-optimize the hyperparameters for the models.
    - Run `1_data_analysis.ipynb` to perform exploratory data analysis.
    - Run `2_octis.ipynb` to perform topic modeling with the OCTIS algorithms and visualize the results.
    - Run `3_BERTopic.ipynb` to perform topic modeling with BERTopic and visualize the results.