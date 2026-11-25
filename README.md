# Flappy Bird AI — Deep Q-Network (DQN)

An experimental Flappy Bird project built with **Python**, **Gymnasium**, and **PyTorch**. A Deep Q-Network (DQN) agent interacts with the `FlappyBird-v0` environment and learns which action to take from the observations and rewards it receives.

The project is intended as a hands-on implementation for learning the fundamentals of **Deep Reinforcement Learning**.

## How it works

The agent learns through repeated interaction with the game:

1. The environment provides the current state (the project enables the environment's LiDAR observations).
2. The agent chooses an action: do nothing or flap.
3. The environment returns the next state, reward, and episode status.
4. The transition is stored in replay memory.
5. The DQN is trained from randomly sampled batches of past experiences.
6. The agent gradually reduces random exploration as training progresses.

The training code uses a policy network and a target network. The target network is periodically synchronised with the policy network to help stabilise learning.

## Features

- Flappy Bird environment using `flappy_bird_gymnasium` and Gymnasium.
- DQN implemented with PyTorch.
- Experience replay using a bounded buffer.
- Epsilon-greedy action selection for exploration and exploitation.
- Configurable hyperparameters in `parameters.yaml`.
- Automatic device selection: Apple Silicon MPS, CUDA, or CPU (depending on availability).
- Model checkpoints and training logs saved in the `runs/` directory.
- A separate script for playing the game manually using the keyboard.

## Project structure

```text
Flappy_Bird_Game/
├── agent.py              # Agent, training loop, and evaluation mode
├── dqn.py                # Deep Q-Network definition
├── experience_replay.py  # Experience replay buffer
├── flappy_bird.py        # Manual keyboard-controlled game
├── parameters.yaml       # DQN hyperparameters
└── runs/                 # Generated model checkpoints and logs
```

## Requirements

- Python 3
- PyTorch
- Gymnasium
- flappy_bird_gymnasium
- PyYAML
- Pygame (used by the rendered/manual game)

## Installation

Clone the repository:

```bash
git clone https://github.com/souravkr0715/Flappy_Bird_Game.git
cd Flappy_Bird_Game
```

Create and activate a virtual environment (recommended):

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install torch gymnasium flappy_bird_gymnasium pyyaml pygame
```

> **PyTorch note:** The project automatically selects MPS on supported Apple Silicon Macs, CUDA on supported NVIDIA systems, or CPU otherwise. Install the PyTorch build appropriate for your system if the standard `pip install torch` command does not work for your setup.

## Run the project

### 1. Train the AI agent

```bash
python agent.py flappybirdv0 --train
```

The argument `flappybirdv0` selects the configuration with the same name in `parameters.yaml`.

**Important:** The current training loop is continuous; it does not have a command-line episode limit. Stop training manually with `Ctrl + C` when you want to end the run. The best model encountered during training is saved to:

```text
runs/flappybirdv0.pt
```

Best-reward records are appended to:

```text
runs/flappybirdv0.log
```

### 2. Watch the trained agent play

Train the agent first so that the model checkpoint exists, then run:

```bash
python agent.py flappybirdv0
```

Without `--train`, the script loads the saved model and runs the environment in human-rendered mode.

### 3. Play manually

To play the game yourself, run:

```bash
python flappy_bird.py
```

Press **Space** to flap. Close the game window to exit.

## Hyperparameters

The current `flappybirdv0` configuration in `parameters.yaml` uses the following values:

| Parameter | Value | Purpose |
|---|---:|---|
| `epsilon_init` | `1.0` | Initial exploration rate |
| `epsilon_min` | `0.05` | Minimum exploration rate |
| `epsilon_decay` | `0.9995` | Exploration-rate decay after each episode |
| `replay_memory_size` | `100000` | Maximum number of stored experiences |
| `mini_batch_size` | `32` | Number of experiences sampled for an update |
| `network_sync_rate` | `10` | Step threshold used by the code to synchronise the target network |
| `alpha` | `0.001` | Learning rate |
| `gamma` | `0.99` | Discount factor |
| `reward_threshold` | `1000` | Per-episode reward limit used by the training loop |

You can experiment with these values in `parameters.yaml`. Keep the configuration key `flappybirdv0` consistent with the command-line argument passed to `agent.py`.

## Notes

- Training time and performance depend on your hardware and the selected hyperparameters.
- A saved checkpoint means the model was saved by the training code; it does not by itself guarantee that the agent has learned to play well. Evaluate the agent by watching its gameplay and comparing its results over multiple episodes.
- The repository currently provides a Python training/evaluation workflow. A hosted browser demo or web deployment is not included yet.

## Future improvements

- Add a fixed episode limit and command-line options for training duration.
- Plot rewards over time to visualise learning progress.
- Add screenshots or a GIF showing the trained agent.
- Build a small web interface to let visitors watch the trained agent play.

## Author

**Sourav Kumar**

GitHub: [@souravkr0715](https://github.com/souravkr0715)

---

This project is a learning-focused implementation of Deep Reinforcement Learning with DQN.
