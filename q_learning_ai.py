import numpy as np
import pickle
import random
import os
import math
from collections import deque
from minimax_ai import MinimaxAI

class QLearningTicTacToe:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1, game_mode=False):
        self.q_table = {}  # State-action value table for AI
        self.opponent_q_table = {}  # State-action value table for opponent
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.player = "O"  # AI plays as 'O' in the game
        self.replay_buffer = deque(maxlen=10000)
        self.visit_counts = {}
        self.initial_alpha = alpha  # used by adaptive_alpha
        self.training_stats = {'wins': [], 'losses': [], 'draws': []}  # For monitoring progress
        self.minimax_ai = MinimaxAI() # Initialize minimax AI for training
        
        # Try to load existing Q-tables or train new ones
        self.load_q_table()
        if not self.q_table or not self.opponent_q_table:
            print("Pre-training Q-learning agent...")
            self.initialize_with_heuristics()  # Initialize with heuristics first
            self.train_curriculum(5000)  # Train with curriculum learning
            self.save_q_table()
            print("Training complete!")
            
        # With this:
        success = self.load_q_table()
        if not success:
            print("No pre-trained Q-table found. Using empty table.")
            # Initialize with basic values but DON'T train
            self.initialize_with_heuristics()

    def convert_2d_to_1d(self, board_2d):
        """Convert 2D board to 1D format for Q-learning."""
        board_1d = [" "] * 9
        for i in range(3):
            for j in range(3):
                cell = board_2d[i][j]
                if cell is not None:
                    board_1d[i*3 + j] = cell
        return board_1d
    
    def convert_1d_to_2d(self, board_1d):
        """Convert 1D board to 2D format for minimax."""
        board_2d = [[None for _ in range(3)] for _ in range(3)]
        for i in range(9):
            row, col = i // 3, i % 3
            board_2d[row][col] = board_1d[i] if board_1d[i] != " " else None
        return board_2d

    def convert_index_to_coordinates(self, index):
        """Convert 1D index to 2D board coordinates."""
        row = index // 3
        col = index % 3
        return row, col

    def get_state(self, board):
        """Convert board to a tuple (immutable for dictionary key)."""
        if isinstance(board, list) and all(isinstance(row, list) for row in board):
            # Convert 2D board to 1D first if needed
            board = self.convert_2d_to_1d(board)
        return self.get_canonical_state(board)

    def get_valid_moves(self, board):
        """Return a list of valid move indices."""
        return [i for i in range(9) if board[i] == " "]

    def choose_action(self, board, q_table, game_mode=False, use_ucb=False):
        """Choose action using epsilon-greedy strategy."""
        if use_ucb:
            return self.choose_action_ucb(board, q_table)
        
        state = self.get_state(board)
        valid_moves = self.get_valid_moves(board)
        
        if not valid_moves:
            return None
        
        # Use lower epsilon during actual gameplay for less randomness
        explore_rate = 0.05 if game_mode else self.epsilon
        
        if random.uniform(0, 1) < explore_rate:
            return random.choice(valid_moves)  # Explore
        
        # Exploit: Choose best known action
        q_values = [q_table.get((state, move), 0) for move in valid_moves]
        max_q = max(q_values)
        best_moves = [valid_moves[i] for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_moves)

    def update_q_value(self, state, action, reward, next_state, q_table):
        """Update Q-table using Q-learning formula with adaptive learning rate."""
        # Use adaptive learning rate
        alpha = self.adaptive_alpha(state, action)
        
        max_future_q = max([q_table.get((next_state, a), 0) for a in self.get_valid_moves(list(next_state))], default=0)
        current_q = q_table.get((state, action), 0)
        q_table[(state, action)] = current_q + alpha * (reward + self.gamma * max_future_q - current_q)
    
    def count_two_in_a_row(self, board, player):
        """Count how many two-in-a-row positions player has with an empty third position."""
        winning_combinations = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        count = 0
        for combo in winning_combinations:
            player_count = sum(1 for pos in combo if board[pos] == player)
            empty_count = sum(1 for pos in combo if board[pos] == " ")
            if player_count == 2 and empty_count == 1:
                count += 1
        return count
    
    def reward_function(self, board, player):
        """Enhanced reward function with intermediate rewards."""
        winner = self.check_winner(board)
        
        if winner == player:
            return 10, True  # Win
        elif winner == "draw":
            return 1, True   # Draw
        elif winner is not None:
            return -10, True  # Loss
        
        # Add strategic rewards
        score = 0
        
        # Reward for creating two-in-a-row opportunities
        score += self.count_two_in_a_row(board, player) * 0.5
        
        # Penalty for allowing opponent two-in-a-row
        opponent = 'X' if player == 'O' else 'O'
        score -= self.count_two_in_a_row(board, opponent) * 0.7
        
        # Reward for taking center
        if board[4] == player:  # Center position
            score += 0.3
            
        # Small negative reward to encourage shorter games
        score -= 0.1

        done = winner is not None or " " not in board
        return score, done

    def check_winner(self, board):
        """Check if there is a winner or a draw."""
        winning_combinations = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for combo in winning_combinations:
            if board[combo[0]] == board[combo[1]] == board[combo[2]] and board[combo[0]] != " ":
                return board[combo[0]]
        if " " not in board:
            return "draw"
        return None
    
    def train(self, episodes=10000, use_replay=True, use_ucb=True, checkpointing=True, 
              checkpoint_interval=10000, early_stopping=True, patience=5):
        """Train the agent through self-play."""
        # For early stopping
        best_win_rate = 0
        no_improvement_count = 0
        
        # For tracking performance
        window_size = 1000
        wins, losses, draws = 0, 0, 0
        
        for episode in range(episodes):
            board = [" "] * 9
            current_player = "X"
            game_history = []
            
            # Adaptive epsilon decay based on win rate
            if episode % window_size == 0 and episode > 0:
                win_rate = wins / window_size
                self.training_stats['wins'].append(win_rate)
                loss_rate = losses / window_size
                self.training_stats['losses'].append(loss_rate)
                draw_rate = draws / window_size
                self.training_stats['draws'].append(draw_rate)
                
                # Adjust epsilon based on performance
                if win_rate > 0.6:
                    self.epsilon = max(0.01, self.epsilon * 0.9)  # Reduce exploration if doing well
                elif win_rate < 0.3:
                    self.epsilon = min(0.5, self.epsilon * 1.1)  # Increase exploration if doing poorly
                else:
                    self.epsilon = max(0.01, self.epsilon * 0.995)  # Normal decay
                
                # Early stopping check
                if early_stopping:
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    if no_improvement_count >= patience:
                        print(f"Early stopping at episode {episode}, no improvement for {patience} windows")
                        break
                
                # Reset counters
                wins, losses, draws = 0, 0, 0
                
                # Print progress
                print(f"Episode {episode}/{episodes}, Win rate: {win_rate:.2f}, Loss rate: {loss_rate:.2f}, Draw rate: {draw_rate:.2f}")
            
            # Play out the game
            while True:
                state = self.get_state(board)
                
                if current_player == "X":
                    action = self.choose_action(board, self.opponent_q_table, use_ucb=use_ucb)
                else:
                    action = self.choose_action(board, self.q_table, use_ucb=use_ucb)
                
                board[action] = current_player
                next_state = self.get_state(board)
                
                reward, done = self.reward_function(board, current_player)
                game_history.append((state, action, reward, next_state, done, current_player))
                
                # Update stats if game is over
                if reward != 0 or " " not in board:
                    winner = self.check_winner(board)
                    if winner == "O":
                        wins += 1
                    elif winner == "X":
                        losses += 1
                    else:
                        draws += 1
                    break
                
                current_player = "O" if current_player == "X" else "X"
            
            # Learning from the game history
            for state, action, reward, next_state, done, player in game_history:
                if player == "X":
                    self.update_q_value(state, action, -reward, next_state, self.opponent_q_table)
                else:
                    self.update_q_value(state, action, reward, next_state, self.q_table)
                
                # Add to replay buffer
                if use_replay:
                    self.add_to_replay_buffer((state, action, reward, next_state, done, player))
            
            # Learn from replay buffer periodically
            if use_replay and episode % 10 == 0:
                self.learn_from_replay_buffer()
            
            # Save checkpoint periodically
            if checkpointing and episode > 0 and episode % checkpoint_interval == 0:
                self.save_q_table(f"q_table_checkpoint_{episode}.pkl")
    
    def train_against_random(self, episodes=5000):
        """Train against a random opponent."""
        for episode in range(episodes):
            board = [" "] * 9
            current_player = "X"  # 'X' starts
            game_history = []
            
            # Adaptive epsilon decay
            self.epsilon = max(0.01, self.epsilon * 0.999)
            
            while True:
                state = self.get_state(board)
                
                if current_player == "X":  # Random opponent
                    valid_moves = self.get_valid_moves(board)
                    action = random.choice(valid_moves) if valid_moves else None
                else:  # AI agent
                    action = self.choose_action(board, self.q_table)
                
                if action is None:  # No valid moves
                    break
                    
                board[action] = current_player
                next_state = self.get_state(board)
                
                reward, done = self.reward_function(board, current_player)
                
                # Only store AI's moves for learning
                if current_player == "O":
                    game_history.append((state, action, reward, next_state, done))
                
                if done:
                    break  # Game over
                
                current_player = "O" if current_player == "X" else "X"
            
            # Learning phase - only update AI's Q-table
            for state, action, reward, next_state, done in game_history:
                self.update_q_value(state, action, reward, next_state, self.q_table)
                self.add_to_replay_buffer((state, action, reward, next_state, done, "O"))
            
            if episode % 10 == 0:
                self.learn_from_replay_buffer()
    
    def train_against_minimax(self, episodes=5000, depth_limit=2):
        """Train against a minimax opponent with limited depth."""
        for episode in range(episodes):
            board = [" "] * 9
            current_player = "X"  # 'X' starts (minimax player)
            game_history = []
            
            # Adaptive epsilon decay
            self.epsilon = max(0.01, self.epsilon * 0.999)
            
            while True:
                state = self.get_state(board)
                
                if current_player == "X":  # Minimax opponent
                    # Convert 1D board to 2D for minimax
                    board_2d = self.convert_1d_to_2d(board)
                    move_2d = self.minimax_ai.best_move(board_2d)
                    
                    # Handle None return case
                    if move_2d is None:
                        valid_moves = self.get_valid_moves(board)
                        action = random.choice(valid_moves) if valid_moves else None
                    else:
                        # Convert back to 1D index
                        action = move_2d[0] * 3 + move_2d[1]
                else:  # AI agent
                    action = self.choose_action(board, self.q_table)
                
                if action is None:  # No valid moves
                    break
                    
                board[action] = current_player
                next_state = self.get_state(board)
                
                reward, done = self.reward_function(board, current_player)
                
                # Only store AI's moves for learning
                if current_player == "O":
                    game_history.append((state, action, reward, next_state, reward != 0 or " " not in board))
                
                if reward != 0 or " " not in board:
                    break  # Game over
                
                current_player = "O" if current_player == "X" else "X"
            
            # Learning phase - only update AI's Q-table
            for state, action, reward, next_state, done in game_history:
                self.update_q_value(state, action, reward, next_state, self.q_table)
                self.add_to_replay_buffer((state, action, reward, next_state, done, "O"))
            
            if episode % 10 == 0:
                self.learn_from_replay_buffer()

    def play_move(self, board):
        """Make a move during an actual game."""
        action = self.choose_action(board, self.q_table, game_mode=True, use_ucb=False)
        return action
    
    def best_move(self, board_2d):
        """Return the best move as (row, col) coordinates."""
        board_1d = self.convert_2d_to_1d(board_2d)
        action = self.play_move(board_1d)
        return self.convert_index_to_coordinates(action)
    
    def save_q_table(self, filename="q_table.pkl"):
        try:
            with open(filename, "wb") as f:
                pickle.dump((self.q_table, self.opponent_q_table, self.training_stats), f)
            print(f"Q-table saved to {filename}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")
    
    def load_q_table(self, filename="q_table.pkl"):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                if len(data) == 2:
                    self.q_table, self.opponent_q_table = data
                else:
                    self.q_table, self.opponent_q_table, self.training_stats = data
                print(f"Q-table loaded from {filename}")
                return True  # Successfully loaded
        except FileNotFoundError:
            print("No pre-trained Q-table found. Starting fresh.")
            return False  # Failed to load
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            return False  # Failed to load

    def get_canonical_state(self, board):
        """Convert board to canonical form using board symmetry."""
        # Generate all possible rotations and reflections
        transformations = []
        b = board[:]
        
        # Add 90-degree rotations
        for _ in range(4):
            transformations.append(tuple(b))
            # Rotate 90 degrees
            b = [b[6], b[3], b[0], b[7], b[4], b[1], b[8], b[5], b[2]]
        
        # Add reflections
        b = board[:]
        b = [b[2], b[1], b[0], b[5], b[4], b[3], b[8], b[7], b[6]]  # Reflect horizontally
        for _ in range(4):
            transformations.append(tuple(b))
            # Rotate 90 degrees
            b = [b[6], b[3], b[0], b[7], b[4], b[1], b[8], b[5], b[2]]
        
        # Return the lexicographically smallest state
        return min(transformations)

    def add_to_replay_buffer(self, experience):
        """Add experience to replay buffer."""
        self.replay_buffer.append(experience)
            
    def learn_from_replay_buffer(self, batch_size=32):
        """Learn from random experiences in replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return
            
        experiences = random.sample(list(self.replay_buffer), batch_size)
        for state, action, reward, next_state, done, player in experiences:
            # Update appropriate Q-table based on player
            q_table = self.q_table if player == 'O' else self.opponent_q_table
            self.update_q_value(state, action, reward, next_state, q_table)

    def adaptive_alpha(self, state, action):
        """Calculate adaptive learning rate based on visit count."""
        key = (state, action)
        self.visit_counts[key] = self.visit_counts.get(key, 0) + 1
        return max(0.01, self.initial_alpha / (1 + 0.1 * self.visit_counts[key]))

    def train_curriculum(self, episodes=50000):
        """Train using curriculum learning - increasing difficulty."""
        total_episodes = episodes
        # First stage: train against random opponent
        print("Stage 1: Training against random opponent...")
        self.train_against_random(total_episodes // 4)
        
        # Second stage: train against minimax with limited depth
        print("Stage 2: Training against easy minimax...")
        self.train_against_minimax(total_episodes // 4, depth_limit=1)
        
        # Third stage: train against harder minimax
        print("Stage 3: Training against harder minimax...")
        self.train_against_minimax(total_episodes // 4, depth_limit=2)
        
        # Fourth stage: full self-play
        print("Stage 4: Training with full self-play...")
        self.train(total_episodes // 4, use_replay=True, use_ucb=True)
        
        print("Curriculum training complete!")

    def initialize_with_heuristics(self):
        """Initialize Q-table with heuristic values."""
        print("Initializing Q-table with heuristics...")
        # Values for empty board
        empty_board = tuple([" "] * 9)
        
        # Prefer center
        self.q_table[(empty_board, 4)] = 0.3
        
        # Then corners
        for corner in [0, 2, 6, 8]:
            self.q_table[(empty_board, corner)] = 0.2
        
        # Sides are less valuable
        for side in [1, 3, 5, 7]:
            self.q_table[(empty_board, side)] = 0.1
        
        # Set up similar values for opponent_q_table
        self.opponent_q_table = self.q_table.copy()
        
        # Add winning and blocking moves for common positions
        self.initialize_winning_blocking_moves()
        
    def initialize_winning_blocking_moves(self):
        """Initialize Q-values for winning and blocking moves."""
        winning_combinations = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        
        # Generate board states with two marks in a row and initialize high Q-values for the winning move
        for player in ["O", "X"]:
            opponent = "X" if player == "O" else "O"
            q_table = self.q_table if player == "O" else self.opponent_q_table
            
            for combo in winning_combinations:
                for empty_pos in combo:
                    other_pos = [pos for pos in combo if pos != empty_pos]
                    
                    # Create a board with two marks in a row and initialize high Q-value for winning move
                    board = [" "] * 9
                    for pos in other_pos:
                        board[pos] = player
                    
                    state = self.get_canonical_state(board)
                    q_table[(state, empty_pos)] = 0.9  # High value for completing a win
                    
                    # Create a board with two opponent marks in a row and initialize high Q-value for blocking move
                    board = [" "] * 9
                    for pos in other_pos:
                        board[pos] = opponent
                    
                    state = self.get_canonical_state(board)
                    q_table[(state, empty_pos)] = 0.7  # High value for blocking opponent's win

    def choose_action_ucb(self, board, q_table, game_mode=False):
        """Choose action using UCB exploration strategy."""
        state = self.get_state(board)
        valid_moves = self.get_valid_moves(board)
        
        if not valid_moves:
            return None
        
        # In actual gameplay, use greedy strategy
        if game_mode:
            q_values = [q_table.get((state, move), 0) for move in valid_moves]
            max_q = max(q_values)
            best_moves = [valid_moves[i] for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_moves)
        
        # During training, use UCB
        total_visits = sum(self.visit_counts.get((state, move), 0) + 1 for move in valid_moves)
        ucb_values = []
        
        for move in valid_moves:
            q_value = q_table.get((state, move), 0)
            visit_count = self.visit_counts.get((state, move), 0) + 1
            exploration = 2.0 * math.sqrt(math.log(total_visits + 1) / visit_count)
            ucb_values.append(q_value + exploration)
        
        best_move = valid_moves[ucb_values.index(max(ucb_values))]
        return best_move

    def visualize_q_values(self, board):
        """Visualize Q-values for current board state."""
        state = self.get_state(board)
        results = []
        
        for i in range(9):
            if board[i] == " ":
                q_value = self.q_table.get((state, i), 0)
                results.append((i, q_value))
            else:
                results.append((i, None))
        
        # Print board with Q-values
        print("\nQ-Values for current board state:")
        for i in range(0, 9, 3):
            row = []
            for j in range(3):
                idx = i + j
                if board[idx] != " ":
                    row.append(f" {board[idx]} ")
                else:
                    q = results[idx][1]
                    row.append(f"{q:.2f}" if q is not None else "   ")
            print("|".join(row))
            if i < 6:
                print("-" * 11)

    def evaluate_agent(self, num_games=1000, opponent_type="random"):
        """Evaluate the agent against different opponents."""
        wins, losses, draws = 0, 0, 0
        
        for _ in range(num_games):
            board = [" "] * 9
            current_player = "X"  # 'X' starts
            
            while True:
                if current_player == "X":  # Opponent
                    if opponent_type == "random":
                        valid_moves = self.get_valid_moves(board)
                        action = random.choice(valid_moves) if valid_moves else None
                    elif opponent_type == "minimax":
                        action = self.minimax_move(board, 3)
                    else:  # Self-play
                        action = self.choose_action(board, self.opponent_q_table, game_mode=True)
                else:  # Our agent
                    action = self.choose_action(board, self.q_table, game_mode=True)
                
                if action is None:  # No valid moves
                    break
                    
                board[action] = current_player
                
                winner = self.check_winner(board)
                if winner is not None:
                    if winner == "O":
                        wins += 1
                    elif winner == "X":
                        losses += 1
                    else:
                        draws += 1
                    break
                
                current_player = "O" if current_player == "X" else "X"
        
        win_rate = wins / num_games
        loss_rate = losses / num_games
        draw_rate = draws / num_games
        
        print(f"Evaluation against {opponent_type} opponent:")
        print(f"Win rate: {win_rate:.2f}")
        print(f"Loss rate: {loss_rate:.2f}")
        print(f"Draw rate: {draw_rate:.2f}")
        
        return win_rate, loss_rate, draw_rate

    def plot_training_progress(self):
        """Plot training progress over time."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.training_stats['wins']:
                print("No training statistics available.")
                return
            
            episodes = range(len(self.training_stats['wins']))
            
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, self.training_stats['wins'], 'g-', label='Wins')
            plt.plot(episodes, self.training_stats['losses'], 'r-', label='Losses')
            plt.plot(episodes, self.training_stats['draws'], 'b-', label='Draws')
            
            plt.title('Q-Learning Agent Training Progress')
            plt.xlabel('Training Windows (1000 episodes each)')
            plt.ylabel('Rate')
            plt.legend()
            plt.grid(True)
            
            plt.savefig('training_progress.png')
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Install it to visualize training progress.")