import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os
from utils import get_model_path, get_plot_path

class DQN(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, output_size=9):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

class DQNTicTacToe:
    def __init__(self, alpha=0.001, gamma=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, buffer_size=10000, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.training_stats = {'wins': [], 'losses': [], 'draws': []}  # For monitoring progress
        self.player = "O"
        self.load_model()  # Load existing model if available

    def set_player(self, player):
        """Set the player marker (X or O)."""
        self.player = player
        self.opponent = 'X' if player == 'O' else 'O'
        
    def get_state(self, board):
        """Convert board to tensor representation for neural network from player perspective."""
        # Get canonical (symmetric) state first
        if isinstance(board[0], list):
            # Convert 2D board to 1D for canonicalization
            board_1d = [" " if cell is None else cell for row in board for cell in row]
            canonical_board = self.get_canonical_state(board_1d)
        else:
            canonical_board = self.get_canonical_state(board)
            
        # Convert to tensor representation
        # Represent from the player's perspective: player is always 1, opponent is -1
        return torch.tensor([
            1.0 if cell == self.player else 
        -1.0 if cell != " " else 
            0.0 for cell in canonical_board
        ], dtype=torch.float32, device=self.device).unsqueeze(0)
    
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
    
    def get_valid_moves(self, board):
        return [i for i in range(9) if board[i] == " "]
    
    def choose_action(self, board, game_mode=False):
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return None
            
        # Use lower epsilon during actual gameplay for less randomness
        explore_rate = 0.0 if game_mode else self.epsilon

        if random.uniform(0, 1) < explore_rate:
            return random.choice(valid_moves)  # Explore
        
        state = self.get_state(board)
        q_values = self.policy_net(state).detach().cpu().numpy()[0]
        valid_q_values = [(q_values[i], i) for i in valid_moves]
        return max(valid_q_values)[1]  # Best valid move
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Ensure consistent tensor dimensions 
        states = torch.cat(states)  # Use cat instead of stack since these are already unsqueezed
        next_states = torch.cat(next_states)
        
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.bool)
        
        # Get Q-values for actions taken
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Get max Q-values for next states
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self, episodes=50000, target_update=1000):
        wins, losses, draws = 0, 0, 0
        stats_window = 1000  # Number of episodes between stats updates

        for episode in range(episodes):
            board = [" "] * 9
            state = self.get_state(board)
            done = False

            while not done:
                action = self.choose_action(board)
                board[action] = self.player  # Use self.player instead of hardcoded "X"
                next_state = self.get_state(board)
                reward, done = self.reward_function(board)

                # Track game outcome
                if done:
                    winner = self.check_winner(board)
                    if winner == self.player:
                        wins += 1
                    elif winner == "draw":
                        draws += 1
                    else:
                        losses += 1

                    self.store_experience(state, action, reward, next_state, done)
                    self.train_step()
                    state = next_state
            
            if episode % target_update == 0:
                self.update_target_network()

            # Calculate and report statistics every 1000 episodes
            if episode % stats_window == 0 and episode > 0:
                win_rate = wins / stats_window
                loss_rate = losses / stats_window
                draw_rate = draws / stats_window
                
                # Store stats
                self.training_stats['wins'].append(win_rate)
                self.training_stats['losses'].append(loss_rate)
                self.training_stats['draws'].append(draw_rate)
                            
                # Print current stats
                print(f"Episode: {episode}/{episodes}, Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}, Epsilon: {self.epsilon:.3f}")
                
                # Reset counters
                wins, losses, draws = 0, 0, 0
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
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

    def reward_function(self, board):
        """Enhanced reward function with intermediate rewards."""
        winner = self.check_winner(board)
        
        if winner == self.player:
            return 10, True  # Win
        elif winner == "draw":
            return 1, True   # Draw
        elif winner is not None:
            return -10, True  # Loss
        
        # Add strategic rewards
        score = 0
        
        # Reward for creating two-in-a-row opportunities
        score += self.count_two_in_a_row(board, self.player) * 0.5
        
        # Penalty for allowing opponent two-in-a-row
        opponent = 'X' if self.player == 'O' else 'O'
        score -= self.count_two_in_a_row(board, opponent) * 0.7
        
        # Reward for taking center
        if board[4] == self.player:  # Center position
            score += 0.3
            
        # Small negative reward to encourage shorter games
        score -= 0.1
        
        done = winner is not None or " " not in board
        return score, done

    def check_winner(self, board):
        winning_combinations = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for combo in winning_combinations:
            if board[combo[0]] == board[combo[1]] == board[combo[2]] and board[combo[0]] != " ":
                return board[combo[0]]
        
        if " " not in board:
            return "draw"
        return None
    
    def play_move(self, board):
        """Make a move during an actual game."""
        return self.choose_action(board, game_mode=True)
    
    def save_model(self, filename=None):
        # Save model weights
        model_path = get_model_path("dqn") if filename is None else filename
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), model_path)
        
        # Save training stats
        stats_filename = model_path.replace('.pth', '_stats.pkl')
        try:
            import pickle
            with open(stats_filename, 'wb') as f:
                pickle.dump(self.training_stats, f)
        except Exception as e:
            print(f"Error saving training stats: {e}")
    
    def load_model(self, filename=None):
        try:
            model_path = get_model_path("dqn") if filename is None else filename
            if os.path.exists(model_path):
                self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
                self.update_target_network()
                
                # Try to load training stats
                stats_filename = model_path.replace('.pth', '_stats.pkl')
                if os.path.exists(stats_filename):
                    import pickle
                    with open(stats_filename, 'rb') as f:
                        self.training_stats = pickle.load(f)
                
                print(f"DQN model loaded from {model_path}")
                return True
            else:
                print(f"Model file {model_path} not found, using untrained model.")
                return False
        except Exception as e:
            print(f"Error loading DQN model: {e}")
            return False

    def best_move(self, board_2d):
        """Return the best move as (row, col) coordinates."""
        # Convert 2D board to 1D
        board_1d = [" " if cell is None else cell for row in board_2d for cell in row]
        
        # If the board is full, return a safe default
        if " " not in board_1d:
            return 0, 0
        
        # Get action using gameplay mode (deterministic)
        action = self.play_move(board_1d)
        
        # Convert 1D action index to 2D board coordinates
        row = action // 3
        col = action % 3
        
        return row, col
            
    def train_curriculum(self, episodes=50000):
        """Train using curriculum learning - increasing difficulty."""
        total_episodes = episodes
        # First stage: train against random opponent
        print("Stage 1: Training against random opponent...")
        self.train_against_random(total_episodes // 4)
        
        # Second stage: train against minimax with limited depth
        print("Stage 2: Training against easy minimax...")
        self.train_against_minimax(total_episodes // 4, depth_limit=2)
        
        # Third stage: train against harder minimax
        print("Stage 3: Training against harder minimax...")
        self.train_against_minimax(total_episodes // 4, depth_limit=4)
        
        # Fourth stage: full self-play with DQN
        print("Stage 4: Training with full self-play...")
        self.train(total_episodes // 4)
        
        print("Curriculum training complete!")

    def train_against_random(self, episodes=5000):
        """Train against a random opponent."""
        opponent_player = 'X' if self.player == 'O' else 'O'
        wins, losses, draws = 0, 0, 0
        stats_window = 1000  # Number of episodes between stats updates

        for episode in range(episodes):
            board = [" "] * 9
            state = self.get_state(board)
            done = False
            
            while not done:
                # DQN agent's turn (X)
                action = self.choose_action(board)
                board[action] = self.player
                reward, done = self.reward_function(board)
                
                # Check if game ended after agent's move
                if done:
                    winner = self.check_winner(board)
                    if winner == self.player:
                        wins += 1
                    elif winner == "draw":
                        draws += 1
                    else:
                        losses += 1

                if not done:
                    # Random opponent's turn (O)
                    valid_moves = self.get_valid_moves(board)
                    opponent_action = random.choice(valid_moves)
                    board[opponent_action] = opponent_player
                    reward, done = self.reward_function(board)

                    # Check if game ended after opponent's move
                    if done:
                        winner = self.check_winner(board)
                        if winner == self.player:
                            wins += 1
                        elif winner == "draw":
                            draws += 1
                        else:
                            losses += 1
                            
                    # Invert reward since this was opponent's move
                    reward = -reward
                
                next_state = self.get_state(board)
                self.store_experience(state, action, reward, next_state, done)
                self.train_step()
                state = next_state
            
            if episode % 100 == 0:
                self.update_target_network()
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Calculate and report statistics every 1000 episodes
            if episode % stats_window == 0 and episode > 0:
                win_rate = wins / stats_window
                loss_rate = losses / stats_window
                draw_rate = draws / stats_window
                
                # Store stats
                self.training_stats['wins'].append(win_rate)
                self.training_stats['losses'].append(loss_rate)
                self.training_stats['draws'].append(draw_rate)
                            
                # Print current stats
                print(f"Episode: {episode}/{episodes}, Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}, Epsilon: {self.epsilon:.3f}")
                
                # Reset counters
                wins, losses, draws = 0, 0, 0

    def convert_1d_to_2d(self, board_1d):
            """Convert 1D board to 2D format for minimax."""
            board_2d = [[None for _ in range(3)] for _ in range(3)]
            for i in range(9):
                row, col = i // 3, i % 3
                board_2d[row][col] = board_1d[i] if board_1d[i] != " " else None
            return board_2d
    
    def train_against_minimax(self, episodes=5000, depth_limit=1):
        """Train against a minimax opponent with limited depth."""
        opponent_player = 'X' if self.player == 'O' else 'O'
        if not hasattr(self, 'minimax_ai'):
            from minimax_ai import MinimaxAI
            self.minimax_ai = MinimaxAI()
        
        wins, losses, draws = 0, 0, 0
        stats_window = 1000  # Number of episodes between stats updates

        for episode in range(episodes):
            board = [" "] * 9
            state = self.get_state(board)
            done = False
            
            while not done:
                # DQN agent's turn
                action = self.choose_action(board)
                board[action] = self.player
                reward, done = self.reward_function(board)
                
                # Check if game ended after agent's move
                if done:
                    winner = self.check_winner(board)
                    if winner == self.player:
                        wins += 1
                    elif winner == "draw":
                        draws += 1
                    else:
                        losses += 1

                if not done:
                    # Minimax opponent's turn
                    board_2d = self.convert_1d_to_2d(board)
                    # Set explicit player for minimax with proper error handling
                    try:
                        if hasattr(self.minimax_ai, 'set_player'):
                            self.minimax_ai.set_player(opponent_player)
                    except Exception as e:
                        print(f"Warning: Could not set player for minimax AI: {e}")
                    
                    move_2d = self.minimax_ai.best_move(board_2d, depth_limit)
                    
                    # Handle case where minimax returns None
                    if move_2d is None:
                        valid_moves = self.get_valid_moves(board)
                        opponent_action = random.choice(valid_moves) if valid_moves else None
                    else:
                        opponent_action = move_2d[0] * 3 + move_2d[1]
                    
                    if opponent_action is not None:
                        board[opponent_action] = opponent_player
                        reward, done = self.reward_function(board)

                        # Check if game ended after opponent's move
                        if done:
                            winner = self.check_winner(board)
                            if winner == self.player:
                                wins += 1
                            elif winner == "draw":
                                draws += 1
                            else:
                                losses += 1
                                
                        # Invert reward since this was opponent's move
                        reward = -reward
                
                next_state = self.get_state(board)
                self.store_experience(state, action, reward, next_state, done)
                self.train_step()
                state = next_state
            
            if episode % 100 == 0:
                self.update_target_network()
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Calculate and report statistics every 1000 episodes
            if episode % stats_window == 0 and episode > 0:
                win_rate = wins / stats_window
                loss_rate = losses / stats_window
                draw_rate = draws / stats_window
                
                # Store stats
                self.training_stats['wins'].append(win_rate)
                self.training_stats['losses'].append(loss_rate)
                self.training_stats['draws'].append(draw_rate)
                            
                # Print current stats
                print(f"Episode: {episode}/{episodes}, Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}, Epsilon: {self.epsilon:.3f}")
                
                # Reset counters
                wins, losses, draws = 0, 0, 0

    def plot_training_progress(self):
        """Plot training progress over time with improved visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.interpolate import make_interp_spline
            
            if not self.training_stats['wins']:
                print("No training statistics available.")
                return
            
            # Create figure with larger size and better resolution
            plt.figure(figsize=(12, 7), dpi=100)
            
            # Get episodes array
            episodes = np.array(range(len(self.training_stats['wins']))) * 1000  # Actual episode numbers
            
            # Plot each metric with interpolated smooth lines
            metrics = ['wins', 'losses', 'draws']
            colors = ['green', 'red', 'blue']
            labels = ['Wins', 'Losses', 'Draws']
            
            for metric, color, label in zip(metrics, colors, labels):
                # Convert to numpy array
                values = np.array(self.training_stats[metric])
                
                # Create smooth line
                X_smooth = np.linspace(episodes.min(), episodes.max(), 300)
                spl = make_interp_spline(episodes, values, k=3)
                Y_smooth = spl(X_smooth)
                
                # Plot smooth line with points
                plt.plot(X_smooth, Y_smooth, color=color, alpha=0.8, label=label)
                plt.scatter(episodes, values, color=color, alpha=0.4, s=30)
            
            plt.title('DQN Agent Learning Progress', fontsize=14, pad=20)
            plt.xlabel('Training Episodes', fontsize=12)
            plt.ylabel('Rate', fontsize=12)
            
            # Customize grid
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Customize legend
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            # Add some padding to the layout
            plt.tight_layout()
            
            # Save with high DPI
            plt.savefig(get_plot_path("dqn"), dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError as e:
            print(f"Required plotting libraries not available: {e}")
            print("Install matplotlib and scipy for visualization.")