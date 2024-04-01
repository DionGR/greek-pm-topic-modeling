import string
from typing import List, Union


class GreekPreprocessing:
    
    def __init__(self, lowercase: bool = True,
        min_df: float = 0.0, max_df: float = 1.0,
        remove_punctuation: bool = True, punctuation: str = string.punctuation,
        remove_numbers: bool = True, lemmatize: bool = False, stem: bool = False,
        stopword_list: Union[str, List[str]] = None, min_chars: int = 2, split: bool = True,
        verbose: bool = False, num_processes: int = None,
        ):
        
        self.lowercase = lowercase
        self.min_df = min_df
        self.max_df = max_df
        self.remove_punctuation = remove_punctuation
        self.punctuation = punctuation
        self.remove_numbers = remove_numbers
        self.lemmatize = lemmatize
        self.stem = stem
        
        if stopword_list is None:
            self.stopwords = []
        elif isinstance(stopword_list, str):
            