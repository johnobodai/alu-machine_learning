#!/bin/bash

# List of filenames
files=(
    "0-neuron.py" "1-neuron.py" "2-neuron.py" "3-neuron.py" "4-neuron.py" 
    "5-neuron.py" "6-neuron.py" "7-neuron.py" "8-neural_network.py" 
    "9-neural_network.py" "10-neural_network.py" "11-neural_network.py" 
    "12-neural_network.py" "13-neural_network.py" "14-neural_network.py" 
    "15-neural_network.py" "16-deep_neural_network.py" "17-deep_neural_network.py" 
    "18-deep_neural_network.py" "19-deep_neural_network.py" "20-deep_neural_network.py" 
    "21-deep_neural_network.py" "22-deep_neural_network.py" "23-deep_neural_network.py" 
    "24-one_hot_encode.py" "25-one_hot_decode.py" "26-deep_neural_network.py" 
    "27-deep_neural_network.py" "28-deep_neural_network.py"
)

# Create files
for file in "${files[@]}"; do
    touch "$file"
done

# Add shebang line to each file
for file in "${files[@]}"; do
    echo '#!/usr/bin/env python3' > tmpfile
    cat "$file" >> tmpfile
    mv tmpfile "$file"
done

