import os

# Define the directory path
directory_path = "/home/john/Documents/ML/alu-machine_learning/unsupervised_learning/autoencoders"
os.makedirs(directory_path, exist_ok=True)  # Create the directory if it doesn't exist

# File names to create
file_names = [
    "0-vanilla.py",
    "1-sparse.py",
    "2-convolutional.py",
    "3-variational.py"
]

# Shebang line to add
shebang_line = "#!/usr/bin/env python3\n"

# Create each file and write the shebang line
for file_name in file_names:
    with open(os.path.join(directory_path, file_name), 'w') as file:
        file.write(shebang_line)

