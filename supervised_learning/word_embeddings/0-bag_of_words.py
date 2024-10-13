#!/usr/bin/env python3
"""Module for generating a bag of words embedding matrix."""


from sklearn.feature_extraction.text import CountVectorizer

def create_bag_of_words(sentences, vocabulary=None):
    """
    Generates a bag of words embedding matrix.

    Args:
        sentences (list): A list of sentences for analysis.
        vocabulary (list, optional): A list of words to consider for the analysis. 
                                      If None, all words from the sentences will be used.

    Returns:
        numpy.ndarray: An array of shape (n_sentences, n_features) containing the embeddings.
        list: A list of the features used for embeddings.
    """
    # Initialize CountVectorizer with the specified vocabulary
    vectorizer = CountVectorizer(vocabulary=vocabulary)

    # Fit the sentences and transform them into a matrix of token counts
    count_matrix = vectorizer.fit_transform(sentences)

    # Convert the count matrix to an array
    embedding_matrix = count_matrix.toarray()

    # Retrieve the feature names (vocabulary words used)
    features = vectorizer.get_feature_names_out().tolist()

    return embedding_matrix, features
