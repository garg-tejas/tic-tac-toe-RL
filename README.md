# Coding Tic-Tac-Toe Game with Multiple AI Implementations

## Overview

A Python-based Tic-Tac-Toe game featuring multiple AI implementations including:

- Minimax Algorithm with Alpha-Beta Pruning
- Q-Learning with Experience Replay
- Deep Q-Network (DQN) with Curriculum Learning
- Monte Carlo Tree Search (MCTS) with RAVE and Virtual Loss

The game provides a graphical user interface built with Pygame, allowing players to compete against different AI opponents or play against another human player.

## Key Features

- 🎮 Interactive GUI with smooth animations
- 🤖 Multiple AI implementations
- 📊 Performance tracking and visualization
- 💾 Persistent score tracking
- ↩️ Undo/Redo move functionality
- 🎯 Pre-trained models included
- 🔄 Training visualization tools

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Virtual environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/garg-tejas/tic-tac-toe-RL.git
cd tic-tac-toe-RL
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```plaintext
pygame==2.6.1
torch==2.2.0
numpy==1.26.0
matplotlib==3.8.0
```

### Optional: GPU Support

For GPU acceleration (if available):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure

```plaintext
tic-tac-toe-RL/
├── main.py                      # Main entry point
├── gui.py                       # GUI implementation using Pygame
├── game.py                      # Core game logic
├── minimax_ai.py                # Minimax algorithm implementation
├── q_learning_ai.py             # Q-Learning implementation
├── dqn_ai.py                    # Deep Q-Network implementation
├── mcts_ai.py                   # Monte Carlo Tree Search implementation
├── train_mcts.py                # MCTS training script
├── train_dqn.py                 # DQN training script
├── train_q_agent.py             # Q-Learning training script
├── requirements.txt             # Python dependencies
└── models/                      # Pre-trained model files
    ├── DQN/model.pth            # Trained DQN Model
    ├── MCTS/model.pkl           # Trained MCTS Model
    └── Q_Learning/model.pkl     # Trained Q-Learning Model
```

## Implementation Details

### Game Logic

- 3x3 grid implementation
- Turn-based gameplay
- Win detection for rows, columns, and diagonals
- Draw detection when board is full
- Move validation and state tracking

### AI Implementations

#### 1. Minimax Algorithm

- Depth-first game tree search
- Alpha-beta pruning for optimization
- Configurable search depth
- Immediate win/block move detection

#### 2. Q-Learning

- Experience replay buffer
- Curriculum learning approach
- State symmetry recognition
- UCB1 exploration strategy
- Adaptive learning rate

#### 3. Deep Q-Network (DQN)

- PyTorch neural network implementation
- Replay memory for experience storage
- Separate target network for stability
- Epsilon-greedy exploration strategy
- Curriculum learning implementation

#### 4. Monte Carlo Tree Search (MCTS)

- UCB1 formula for exploration/exploitation balance
- Rapid Action Value Estimation (RAVE) for improved learning
- Virtual loss for search parallelization
- Transposition table for state caching
- Node recycling for memory efficiency
- Curriculum learning training approach

## Usage

### Running the Game

1. Start the game:

```bash
python main.py
```

2. From the main menu, select:
   - Game Mode (Human vs. AI or Human vs. Human)
   - AI Opponent (when playing against AI)
   - Player Marker (X or O)

### Controls

- Mouse Click: Make a move
- Undo Button: Revert last move
- Redo Button: Replay undone move
- New Game Button: Start a fresh game

### Training AI Models

Before playing, you may want to train the AI models:

1. Train the Q-Learning agent:

```bash
python train_q_agent.py
```

2. Train the DQN agent:

```bash
python train_dqn.py
```

3. Train the MCTS agent:

```bash
python train_mcts.py
```

Training progress and statistics will be saved automatically.

### Pre-trained Models

The repository includes pre-trained models:

- `models/q_learning/model.pkl`: Pre-trained Q-Learning model
- `models/dqn/model.pth`: Pre-trained DQN model
- `models/mcts/model.pkl`: Pre-trained MCTS model

These will be loaded automatically when starting the game.

### Visualization

Training progress can be visualized:

- Win/Loss/Draw rates over time
- Q-value heatmaps
- Learning curves

## Advanced Features and Customization

### Model Configuration

#### DQN Parameters

```python
# Customize DQN hyperparameters in train_dqn.py
agent = DQNTicTacToe(
    alpha=0.001,        # Learning rate
    gamma=0.9,          # Discount factor
    epsilon=1.0,        # Initial exploration rate
    epsilon_min=0.1,    # Minimum exploration rate
    epsilon_decay=0.995,# Exploration decay rate
    buffer_size=10000,  # Replay buffer size
    batch_size=32       # Training batch size
)
```

#### Q-Learning Parameters

```python
# Customize Q-Learning parameters in train_q_agent.py
agent = QLearningTicTacToe(
    alpha=0.1,          # Learning rate
    gamma=0.95,         # Discount factor
    epsilon=0.1,        # Exploration rate
    game_mode=False     # Training mode
)
```

#### MCTS Parameters

```python
# Customize MCTS parameters in train_mcts.py
agent = MCTSAI(
    exploration_weight=1.41,  # UCB exploration parameter
    time_limit=0.5,          # Search time per move (seconds)
    temperature=0.1          # Temperature for move selection
)
```

### Custom Training Settings

#### Curriculum Learning

```python
# Adjust curriculum learning stages
TOTAL_EPISODES = 20000
RANDOM_STAGE = TOTAL_EPISODES // 4    # 25% against random
MINIMAX_STAGE = TOTAL_EPISODES // 4   # 25% against minimax
SELF_PLAY = TOTAL_EPISODES // 2       # 50% self-play
```

#### Experience Replay

```python
# Configure experience replay in q_learning_ai.py
REPLAY_BUFFER_SIZE = 10000
REPLAY_BATCH_SIZE = 32
REPLAY_FREQUENCY = 10  # Learn from replay every 10 episodes
```

## Troubleshooting and Advanced Features

### Common Issues

#### Model Loading Errors

```bash
# If you see "Model file not found" errors:
mkdir models  # Create models directory if missing
python train_q_agent.py  # Train new Q-learning model
python train_dqn.py     # Train new DQN model
python train_mcts.py    #Train new MCTS model
```

#### GPU/CUDA Issues

```bash
# If you encounter CUDA errors, force CPU usage:
export CUDA_VISIBLE_DEVICES=""  # On Linux
```

#### Memory Issues

```python
# Adjust replay buffer size in dqn_ai.py
buffer_size=5000  # Reduce if RAM usage is too high
```

### Advanced Training Options

#### Custom Training Curriculum

```python
# Modify curriculum stages in train_curriculum()
agent.train_against_random(episodes=2000)     # Stage 1
agent.train_against_minimax(episodes=2000)    # Stage 2
agent.train(episodes=2000)                    # Stage 3
```

#### Early Stopping Configuration

```python
# Configure early stopping in training scripts
agent.train(
    episodes=20000,
    early_stopping=True,
    patience=5,          # Stop after 5 windows without improvement
    window_size=1000     # Evaluation window size
)
```

### Performance Monitoring

#### Training Progress

- Training statistics are saved automatically
- View progress plots in `training_progress.png`
- Monitor win rates during training
- Track exploration vs exploitation balance

#### Model Evaluation

```python
# Evaluate trained models against different opponents
agent.evaluate_agent(num_games=1000, opponent_type="random")
agent.evaluate_agent(num_games=1000, opponent_type="minimax")
```

### Code Extensions

#### Adding New AI Methods

1. Create new AI class implementing required methods:
   - `best_move(board_2d)`
   - `set_player(player)`
2. Update GUI to include new AI option
3. Add training script if needed

## Contributing and Future Development

### Planned Features

- 🎮 Additional game modes (Championship, Tournament)
- 🤖 More AI implementations (Alpha-Zero, TD Learning)
- 📊 Interactive training visualization
- 🎯 Difficulty levels for each AI
- 🔄 Real-time learning during gameplay

### Known Issues

1. GUI might freeze briefly during AI's first move
2. Training can be memory-intensive on low-RAM systems
3. Some visual glitches in animation on window resize

### Project Goals

This project was created to:

- Learn and experiment with different AI approaches
- Practice implementing reinforcement learning
- Create a fun, interactive game with AI opponents
- Understand the trade-offs between different AI methods

### Local Development

Feel free to fork and experiment! Some ideas:

- Try different neural network architectures
- Add your own AI implementations
- Improve the GUI and animations
- Optimize the training process

## Acknowledgments

- Built with PyGame and PyTorch
- Inspired by various RL tutorials and guides
- Special thanks to the open source community

## Contact

For questions or suggestions, feel free to:

- Open an issue
- Submit a pull request
- Contact me directly

Happy gaming and AI experimenting! 🎮🤖
