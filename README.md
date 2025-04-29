# DQN-Maze-Path-Finding


A reinforcement learning approach to pathfinding with predefined start and goal points using Deep Q-Networks (DQN).

![Pathfinding Demo](./Images)

## Overview

This project implements a Deep Q-Network (DQN) to solve pathfinding problems in various environments. The agent learns to navigate from a specified starting point to a goal point efficiently by interacting with the environment and receiving rewards.

### Key Features

- Implementation of DQN with experience replay and target networks
- Customizable grid-based environments with obstacles
- Visualization tools for pathfinding progress and learned policies
- Configurable reward functions and hyperparameters
- Pre-trained models for quick demonstration
- Support for different map types and complexity levels

## Project Structure

```
dqn-pathfinding/
├── agents/                  # DQN agent implementation
│   ├── dqn_agent.py         # DQN agent class
│   └── memory.py            # Experience replay buffer
├── environments/            # Environment implementations
│   ├── grid_world.py        # Grid world environment
│   └── obstacles.py         # Obstacle generation utilities
├── models/                  # Neural network models
│   ├── dqn_model.py         # DQN model architecture
│   └── saved/               # Saved model weights
├── utils/
│   ├── visualization.py     # Visualization utilities
│   └── config.py            # Configuration utilities
├── maps/                    # Map configuration files
├── train.py                 # Main training script
├── demo.py                  # Demo script for visualization
├── evaluate.py              # Evaluation script
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Environment

The environment is a grid world where:
- The agent starts at a specified initial position
- The goal is at a specified target position
- Obstacles may be present in the grid
- The agent can move in four directions: up, down, left, right
- The agent receives rewards based on its actions (reaching the goal, hitting obstacles, etc.)

## DQN Implementation

Our DQN implementation includes:

1. **Deep Q-Network**: Neural network that predicts Q-values for each action
2. **Experience Replay**: Buffer to store and sample experiences for training
3. **Target Network**: Separate network for stable Q-value targets
4. **Epsilon-Greedy Exploration**: Balance between exploration and exploitation

## Results

After training, the agent learns to find the optimal or near-optimal path from the start to the goal position. Performance metrics include:

- Average steps per episode
- Success rate
- Average reward
- Q-value convergence

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym for inspiration on environment design
- DeepMind's DQN paper and implementation
- Contributors and maintainers of the PyTorch and TensorFlow libraries
