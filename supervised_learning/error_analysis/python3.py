import os

# Define the shebang line to be added
shebang = "#!/usr/bin/env python3\n"

# List of files to which the shebang line should be added
files = [
    "0-create_confusion.py",
    "1-sensitivity.py",
    "2-precision.py",
    "3-specificity.py",
    "4-f1_score.py",
    "5-error_handling",
]

# Iterate over each file
for file in files:
    # Read the current content of the file
    with open(file, 'r') as f:
        content = f.read()
    
    # Check if the shebang line is already present
    if not content.startswith(shebang):
        # Add the shebang line to the top of the content
        content = shebang + content
        
        # Write the updated content back to the file
        with open(file, 'w') as f:
            f.write(content)

print("Shebang lines added to all specified files.")

