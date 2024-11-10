### README.md

markdown
# Q-Learning Agent for FrozenLake Environment

This project implements a Q-learning agent that is trained to navigate the FrozenLake environment, a standard reinforcement learning task provided by OpenAI Gym. The agent learns to find the best path to the goal while avoiding holes in the frozen lake.

## Project Overview

The project includes the following key components:
1. **Q-Table Initialization**: Initializes the Q-table with zeros.
2. **Epsilon-Greedy Strategy**: Implements the epsilon-greedy algorithm to balance exploration and exploitation.
3. **Q-Learning**: Implements the Q-learning algorithm to update the Q-table during training.
4. **Play Function**: Allows the trained agent to play an episode, choosing actions based on the learned Q-table.

## File Structure


reinforcement_learning/
    └── q_learning/
        ├── 0-load_env.py
        ├── 1-q_init.py
        ├── 2-epsilon_greedy.py
        ├── 3-q_learning.py
        ├── 4-play.py
        ├── 1-main.py
        ├── 2-main.py
        ├── 3-main.py
        ├── 4-main.py


## Description of Files

### `0-load_env.py`
Contains the function `load_frozen_lake` to create and load the FrozenLake environment. The environment can be customized by changing parameters such as map size and slipperiness.

### `1-q_init.py`
Contains the function `q_init` to initialize the Q-table as a numpy array filled with zeros. The shape of the Q-table depends on the environment's state and action space.

### `2-epsilon_greedy.py`
Contains the function `epsilon_greedy`, which implements the epsilon-greedy algorithm to choose an action. It explores randomly with probability `epsilon` and exploits the learned Q-table with probability `1 - epsilon`.

### `3-q_learning.py`
Contains the function `train`, which implements the Q-learning algorithm. It updates the Q-table through interactions with the environment, using a learning rate (`alpha`), discount factor (`gamma`), and epsilon decay for exploration.

### `4-play.py`
Contains the function `play`, which allows the trained agent to play an episode in the environment. The agent selects actions by exploiting the learned Q-values and the environment is rendered at each step.

### Main Files
- `1-main.py`: Tests the Q-table initialization.
- `2-main.py`: Tests the epsilon-greedy function.
- `3-main.py`: Trains the agent using Q-learning and prints out the rewards.
- `4-main.py`: Plays a game using the trained Q-table and visualizes the environment step by step.

## Requirements

To run the project, you will need to install the following dependencies:

- Python 3.x
- NumPy
- Gym (for the FrozenLake environment)

You can install the required libraries using `pip`:

bash
pip install numpy gym


## Usage

1. **Initialize Q-table**:
    To initialize the Q-table for the environment:

    bash
    $ ./1-main.py
    

2. **Test Epsilon-Greedy**:
    To test the epsilon-greedy action selection:

    bash
    $ ./2-main.py
    

3. **Train the Q-learning Agent**:
    To train the agent using Q-learning:

    bash
    $ ./3-main.py
    

4. **Play the Environment**:
    To let the trained agent play an episode and display the environment:

    bash
    $ ./4-main.py
    
