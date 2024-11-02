#!/usr/bin/env python3


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense


def autoencoder(input_dims, hidden_layers, latent_dims):
  """Creates an autoencoder model.

  Args:
    input_dims: Dimensionality of the input data.
    hidden_layers: A list of integers specifying the number of nodes in each hidden layer of the encoder.
    latent_dims: Dimensionality of the latent space.

  Returns:
    A tuple of three models: encoder, decoder, and autoencoder.
  """

  # Encoder
  inputs = Input(shape=(input_dims,))
  x = inputs
  for layer_size in hidden_layers:
    x = Dense(layer_size, activation='relu')(x)
  latent = Dense(latent_dims, activation='relu')(x)
  encoder = tf.keras.Model(inputs=inputs, outputs=latent)

  # Decoder
  latent_inputs = Input(shape=(latent_dims,))
  x = latent_inputs
  for layer_size in reversed(hidden_layers):
    x = Dense(layer_size, activation='relu')(x)
  outputs = Dense(input_dims, activation='sigmoid')(x)
  decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs)

  # Autoencoder
  auto = tf.keras.Model(inputs=inputs, outputs=decoder(encoder(inputs)))
  auto.compile(optimizer='adam', loss='binary_crossentropy')

  return encoder, decoder,
 auto
