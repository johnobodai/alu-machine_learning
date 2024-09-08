# Project Name

## Overview

Brief description of the project.

## Setup

1. Clone the repository.
2. Navigate to the project directory.
3. Create a virtual environment and activate it.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4. Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```
5. Run the Flask app.
    ```bash
    python src/app.py
    ```

## Usage

### Predict Endpoint

- **URL**: `/predict`
- **Method**: `POST`
- **Payload**:
    ```json
    {
        "model_name": "decision_tree",
        "input_data": { ... }
    }
    ```
- **Response**:
    ```json
    {
        "prediction": [ ... ]
    }
    ```

## Directory Structure

- **`data/`**: Contains training and test data.
- **`models/`**: Contains saved models.
- **`notebook/`**: Contains Jupyter notebooks.
- **`src/`**: Contains source code including the Flask app.

