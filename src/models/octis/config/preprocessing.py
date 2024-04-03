stanza_greek_config = {
	'processors': 'tokenize,pos',
    'lang': 'el',
    "pos_model_path": "models/octis/data/checkpoints/stanza/el_gdt_tagger.pt", 
    "lemma_model_path": "models/octis/data/checkpoints/stanza/el_gdt_lemmatizer.pt",
}

preprocessor_gr_params = {
    "remove_punctuation": True, 
    "remove_numbers": True,
    "max_df": 0.2, 
    "min_df": 0.01, 
    "min_chars": 4,
    "min_words": 20, 
    "num_processes": 6
}
