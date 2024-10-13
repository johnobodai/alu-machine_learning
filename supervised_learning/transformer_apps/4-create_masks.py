import tensorflow as tf

def create_masks(inputs, target):
    """
    Creates all masks for training/validation.
    
    Parameters
    ----------
    inputs : tf.Tensor
        A tensor of shape (batch_size, seq_len_in) that contains the input sentence.
    target : tf.Tensor
        A tensor of shape (batch_size, seq_len_out) that contains the target sentence.
    
    Returns
    -------
    encoder_mask : tf.Tensor
        Padding mask of shape (batch_size, 1, 1, seq_len_in) for the encoder.
    combined_mask : tf.Tensor
        Mask of shape (batch_size, 1, seq_len_out, seq_len_out) used in the 1st attention block of the decoder.
    decoder_mask : tf.Tensor
        Padding mask of shape (batch_size, 1, 1, seq_len_out) for the decoder.
    """
    # Create the encoder mask
    encoder_mask = tf.cast(tf.equal(inputs, 0), tf.float32)  # Assuming padding is represented by 0
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]  # Shape: (batch_size, 1, 1, seq_len_in)

    # Create the decoder padding mask
    decoder_padding_mask = tf.cast(tf.equal(target, 0), tf.float32)  # Assuming padding is represented by 0
    decoder_padding_mask = decoder_padding_mask[:, tf.newaxis, tf.newaxis, :]  # Shape: (batch_size, 1, 1, seq_len_out)

    # Create look ahead mask for the decoder
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len_out, seq_len_out)), -1, 0)  # Upper triangular matrix
    look_ahead_mask = tf.cast(look_ahead_mask, tf.float32)  # Shape: (seq_len_out, seq_len_out)
    
    # Combine the look ahead mask and the decoder padding mask
    combined_mask = tf.maximum(decoder_padding_mask, look_ahead_mask)  # Shape: (batch_size, 1, seq_len_out, seq_len_out)

    return encoder_mask, combined_mask, decoder_padding_mask

