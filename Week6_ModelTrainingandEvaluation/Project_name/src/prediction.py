def predict(model, input_data):
    # Ensure input_data is a 2D array
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    return model.predict(input_data)

