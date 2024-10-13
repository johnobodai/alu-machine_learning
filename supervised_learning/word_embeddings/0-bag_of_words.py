#!/usr/bin/env python3
"""
This module contains a function to create a bag of words embedding matrix.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list): List of sentences to analyze.
        vocab (list, optional): List of vocabulary words to use for the analysis. 
                                If None, all words within sentences will be used.

    Returns:
        numpy.ndarray: Embeddings of shape (s, f) containing the bag of words matrix.
        list: Features used for embeddings.
    """
    # Initialize CountVectorizer with the specified vocabulary
    vectorizer = CountVectorizer(vocabulary=vocab)
    
    # Fit and transform the sentences to create the embeddings
    embeddings = vectorizer.fit_transform(sentences).toarray()
    
    # Get the features (vocabulary words used)
    features = vectorizer.get_feature_names_out().tolist()
    
    return embeddings, features

