import string
from typing import Dict, List, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import stanza
from octis.dataset.dataset import Dataset
from tqdm.contrib.concurrent import process_map

from models.octis.config.preprocessing import stanza_greek_config


pipeline = stanza.Pipeline(**stanza_greek_config)


class GreekStanzaPreprocessor:
    
    def __init__(self, vocabulary: List[str] = None,
                    min_df: float = 0.0, max_df: float = 1.0, min_chars: int = 2, min_words: int = 10,
                    remove_numbers: bool = True, remove_punctuation: bool = True, punctuation: str = string.punctuation,
                    stopword_list: Union[str, List[str]] = None, 
                    num_processes: int = None
        ):
        
        self.min_df = min_df
        self.max_df = max_df
        self.min_chars = min_chars
        self.min_words = min_words
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.punctuation = punctuation
        self.num_processes = num_processes
        self.vocabulary = vocabulary
        self.preprocessing_steps = []
        
        
        self.preprocessing_steps.append("lowercase")    
        self.preprocessing_steps.append("lemmatize")
        self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
        self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
        
        if remove_punctuation:
            self.preprocessing_steps.append("remove_punctuation")    
        
        if stopword_list is not None:
            stopword_string = " ".join(stopword_list)
            stopword_words = pipeline(stopword_string).iter_words()
            stopword_list = [token.lemma for token in stopword_words]
        
        self.stopwords = list(set(stopword_list))
        
        self.vectorizer = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, 
                                lowercase=True,
                                token_pattern=r"(?u)\b[\w|\-]{" + str(self.min_chars) + r",}\b",
                                stop_words=self.stopwords)


    def preprocess_dataset(self, documents_path):
        documents = [line.strip() for line in open(documents_path, 'r').readlines()]
        
        if self.num_processes is not None:
            chunksize = max(1, len(documents) // (self.num_processes * 20))
            documents = process_map(self.preprocess_document, documents, max_workers=self.num_processes, chunksize=chunksize)
        
        if self.vocabulary is None:
            self.vocabulary = self.extract_vocabulary(documents)
                
        final_documents, final_labels, document_indexes = [], [], []
        for i, document in enumerate(documents):
            doc_new = [word for word in document.split() if word in self.vocabulary]
            
            if len(doc_new) >= self.min_words:
                final_documents.append(doc_new)
                document_indexes.append(i)

        metadata =  {"total_documents": len(documents), 
                     "vocabulary_length": len(self.vocabulary),
                     "preprocessing-info": self.preprocessing_steps
                    }
        
        train, test = train_test_split(range(len(final_documents)), test_size=0.15, random_state=1)
        train, validation = train_test_split(train, test_size=0.10, random_state=1)

        metadata["last-training-doc"] = len(train)
        metadata["last-validation-doc"] = len(validation) + len(train)
        
        partitioned_corpus = [final_documents[doc] for doc in train + validation + test]
        document_indexes = [document_indexes[doc] for doc in train + validation + test]
        
        return Dataset(partitioned_corpus, vocabulary=self.vocabulary, metadata=metadata, labels=final_labels,
                        document_indexes=document_indexes)

        
    def extract_vocabulary(self, documents):
        self.vectorizer.fit_transform(documents)
        vocabulary = set(self.vectorizer.get_feature_names_out())
        
        return vocabulary
    
    def preprocess_document(self, document):
        
        doc_pipe = pipeline(document)
        doc_new = [word.lemma for word in doc_pipe.iter_words() 
                                if word.lemma not in self.stopwords and len(word.lemma) >= self.min_chars]
        
        if self.remove_numbers:
            doc_new = [word for word in doc_new if not word.isdigit()]
        
        if self.remove_punctuation:
            doc_new = [word for word in doc_new if word not in self.punctuation]
        
        return " ".join(doc_new)
    
    def get_vectorizer(self):
        return self.vectorizer
    
    def get_vocabulary(self):
        return self.vocabulary