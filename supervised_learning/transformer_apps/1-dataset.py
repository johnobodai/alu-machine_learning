#!/usr/bin/env python3
"""
Class Dataset
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import tokenizers
import numpy as np


class Dataset:
    """Class Dataset for loading and preparing a dataset for machine translation"""
    
    def __init__(self):
        """
        Initializes the Dataset class instance.
        Creates the instance attributes:
        - data_train: the train split of the dataset
        - data_valid: the validation split of the dataset
        - tokenizer_pt: the Portuguese tokenizer
        - tokenizer_en: the English tokenizer
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                      split='train',
                                      as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                      split='validation',
                                      as_supervised=True)
        self.tokenizer_pt = None
        self.tokenizer_en = None

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.
        
        Parameters
        ----------
        data : tf.data.Dataset
            A dataset whose examples are formatted as a tuple (pt, en).
        
        Returns
        -------
        tokenizer_pt : tokenizer
            The Portuguese tokenizer.
        tokenizer_en : tokenizer
            The English tokenizer.
        """
        # Create a list to hold the Portuguese and English sentences
        pt_sentences = []
        en_sentences = []
        
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        # Create sub-word tokenizers
        tokenizer_pt = tokenizers.Tokenizer(tokenizers.models.BPE())
        tokenizer_en = tokenizers.Tokenizer(tokenizers.models.BPE())

        # Train the Portuguese tokenizer
        tokenizer_pt.train_from_iterator(pt_sentences, vocab_size=2**15)

        # Train the English tokenizer
        tokenizer_en.train_from_iterator(en_sentences, vocab_size=2**15)

        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en
        
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens.
        
        Parameters
        ----------
        pt : tf.Tensor
            A tensor containing the Portuguese sentence.
        en : tf.Tensor
            A tensor containing the corresponding English sentence.
        
        Returns
        -------
        pt_tokens : np.ndarray
            An array containing the Portuguese tokens.
        en_tokens : np.ndarray
            An array containing the English tokens.
        """
        # Tokenize sentences and add start and end tokens
        pt_tokens = self.tokenizer_pt.encode(pt.numpy().decode('utf-8')).ids
        en_tokens = self.tokenizer_en.encode(en.numpy().decode('utf-8')).ids
        
        # Append start and end tokens
        pt_tokens = [self.tokenizer_pt.get_vocab_size()] + pt_tokens + [
            self.tokenizer_pt.get_vocab_size() + 1]
        en_tokens = [self.tokenizer_en.get_vocab_size()] + en_tokens + [
            self.tokenizer_en.get_vocab_size() + 1]
        
        return np.array(pt_tokens), np.array(en_tokens)

