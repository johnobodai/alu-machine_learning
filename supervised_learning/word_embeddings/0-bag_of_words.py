#!/usr/bin/env python3
"""Creating a bag of words embedding matrix."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix.

    sentences is a list of sentences to analyze.
    vocab is a list of the vocabulary words to use for the analysis.
    If None, all words within sentences should be used.

    Returns: embeddings, features
    embeddings is a numpy.ndarray of shape (s, f) containing the embeddings.
    s is the number of sentences in sentences.
    f is the number of features analyzed.

    features is a list of the features used for embeddings.
    """

    # Initialize CountVectorizer with the specified vocabulary
    vectorizer = CountVectorizer(vocabulary=vocab)
    
    # Transform the sentences into a count matrix
    count_matrix = vectorizer.fit_transform(sentences)
    
    # Convert the sparse matrix to a dense numpy array
    embeddings = count_matrix.toarray()
    
    # Get the feature names (vocabulary words)
    features = vectorizer.get_feature_names_out().tolist()
    
    return embeddings, features

