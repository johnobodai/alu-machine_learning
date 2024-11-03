### README.md

```markdown
# Reinforcement Learning: Policy Gradients

This project focuses on implementing policy gradients in reinforcement learning, using the `CartPole-v1` environment provided by the OpenAI Gym library. The implementation includes creating and training a simple policy-based agent using Monte Carlo Policy Gradient.

---

## Project Structure

- **`policy_gradient.py`**: Contains the implementation of the `policy` and `policy_gradient` functions.
- **`train.py`**: Contains the implementation of the training function to train the agent using policy gradients.
- **`0-main.py`**: Demonstrates the computation of policy probabilities using the `policy` function.
- **`1-main.py`**: Demonstrates the computation of action probabilities and gradients using the `policy_gradient` function.
- **`2-main.py`**: Demonstrates the training of the agent using the `train` function and plots the scores.
- **`3-main.py`**: Extends `2-main.py` to optionally render the environment during training.

---

## Requirements

This project is implemented using **Python 3.5** and the following dependencies:

- `numpy==1.15`
- `gym==0.7`
- `matplotlib`

**Note**: The environment and library versions must match these requirements due to compatibility considerations.

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/alu-machine_learning.git
   cd alu-machine_learning/reinforcement_learning/policy_gradients
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install numpy==1.15 gym==0.7 matplotlib
   ```

---

## Usage

### Run Example Scripts

1. **Compute Policy Probabilities**:
   ```bash
   python3 0-main.py
   ```

2. **Compute Monte-Carlo Policy Gradient**:
   ```bash
   python3 1-main.py
   ```

3. **Train the Agent**:
   ```bash
   python3 2-main.py
   ```

   This will generate a plot of the agent's scores over episodes.

4. **Train with Rendering**:
   ```bash
   python3 3-main.py
   ```

   This will render the environment every 1000 episodes during training.

---

## Functions

### `policy(matrix, weight)`
- Computes the softmax policy probabilities given the state matrix and weight matrix.

### `policy_gradient(state, weight)`
- Computes the Monte-Carlo policy gradient based on the current state and weight matrix.

### `train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False)`
- Trains the agent using policy gradients.
- Parameters:
  - `env`: The Gym environment.
  - `nb_episodes`: Number of training episodes.
  - `alpha`: Learning rate.
  - `gamma`: Discount factor.
  - `show_result`: If `True`, renders the environment every 1000 episodes.
- Returns: A list of scores (total rewards) for each episode.

---

