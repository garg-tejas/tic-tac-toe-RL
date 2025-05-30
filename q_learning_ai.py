import numpy as np
import pickle
import random
import os
import math
from collections import deque
from minimax_ai import MinimaxAI
from utils import get_model_path, get_plot_path

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
        
        # Try to load existing Q-tables
        success = self.load_q_table()
        if not success:
            print("No pre-trained Q-table found. Initializing with heuristics.")
            self.initialize_with_heuristics()

    def is_terminal(self, board):
        """Check if the game has reached a terminal state."""
        return self.check_winner(board) is not None or " " not in board

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

    def choose_action(self, board, q_table=None, epsilon=None, use_ucb=False, game_mode=False):
        """Choose action using epsilon-greedy or UCB policy."""
        # Use appropriate q_table
        if q_table is None:
            q_table = self.q_table
            
        # Use UCB if specified
        if use_ucb:
            return self.choose_action_ucb(board, q_table, game_mode=game_mode)
            
        # Epsilon-greedy implementation
        if epsilon is None:
            epsilon = self.epsilon
            
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return None
            
        # Use lower epsilon in game mode
        explore_rate = 0.01 if game_mode else epsilon
            
        # Exploration: random move
        if random.random() < explore_rate:
            return random.choice(valid_moves)
            
        # Exploitation: use Q-table
        state_key = self.get_state_key(board)
        if state_key in q_table:
            # Get Q-values for valid moves only
            valid_q_values = [(q_table[state_key][move], move) for move in valid_moves]
            return max(valid_q_values, key=lambda x: x[0])[1]
        else:
            # If state not in Q-table, use expert knowledge or random
            best_move = self.get_expert_move(board, self.player)
            if best_move is not None and best_move in valid_moves:
                return best_move
            return random.choice(valid_moves)

    def update_q_value(self, state, action, reward, next_state, q_table):
        """Update Q-table using Q-learning formula with adaptive learning rate."""
        # Use adaptive learning rate
        alpha = self.adaptive_alpha(state, action)
        
        # Make sure state_key is a consistent format
        state_key = self.get_state_key(state) if not isinstance(state, str) else state
        next_state_key = self.get_state_key(next_state) if not isinstance(next_state, str) else next_state
        
        # Initialize state in q_table if not present
        if state_key not in q_table:
            q_table[state_key] = [0.0] * 9
        
        # Get valid moves in next state
        if isinstance(next_state_key, tuple):
            next_board = list(next_state_key)
        elif isinstance(next_state_key, str) and next_state_key.startswith('('):
            # Try to parse the string representation of a tuple
            import ast
            try:
                next_board = list(ast.literal_eval(next_state_key))
            except:
                next_board = next_state_key
        else:
            next_board = next_state_key
        valid_moves = self.get_valid_moves(next_board)
        
        # Calculate max future Q-value
        max_future_q = 0
        if valid_moves:
            if next_state_key in q_table:
                max_future_q = max([q_table[next_state_key][a] for a in valid_moves])
        
        # Update Q-value
        current_q = q_table[state_key][action]
        q_table[state_key][action] = current_q + alpha * (reward + self.gamma * max_future_q - current_q)
    
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
    
    def pretrain_on_expert_moves(self, num_examples=10000):
        """Pre-train the Q-table with expert Tic-Tac-Toe moves."""
        print("Pre-training Q-learning with expert moves...")
        
        # Track stats
        examples_created = 0
        
        # Generate expert data
        for _ in range(num_examples):
            # Create a random board with 1-5 moves already played
            board = [" "] * 9
            moves_made = random.randint(1, 5)
            current_player = 'X'  # Start with X
            
            # Make random moves to reach a mid-game position
            for _ in range(moves_made):
                valid_moves = self.get_valid_moves(board)
                if not valid_moves:
                    break
                move = random.choice(valid_moves)
                board[move] = current_player
                current_player = 'O' if current_player == 'X' else 'X'
                
                # Stop if game is over
                if self.is_terminal(board):
                    break
            
            # If game is already over, skip this example
            if self.is_terminal(board):
                continue
                
            # Now find the "expert move" for this position
            state_key = self.get_state_key(board)
            best_move = self.get_expert_move(board, current_player)
            
            if best_move is not None:
                # Update Q-table with high value for best move, low for others
                if state_key not in self.q_table:
                    self.q_table[state_key] = [0.0] * 9  # Initialize Q-values for all actions
                    
                # Assign higher values to expert moves
                for action in range(9):
                    if action == best_move:
                        self.q_table[state_key][action] = 5.0  # High value for best action
                    elif board[action] != " ":
                        self.q_table[state_key][action] = -5.0  # Invalid move
                    else:
                        self.q_table[state_key][action] = 0.0  # Neutral for other valid moves
                        
                examples_created += 1
        
        print(f"Generated {examples_created} expert examples for Q-learning")

    def get_state_key(self, board):
        """Convert board to a hashable key for the Q-table."""
        # If the board is already in canonical form, just return it as string
        if isinstance(board, str):
            return board
            
        # Otherwise, convert board to canonical form (handling symmetries)
        return str(self.get_canonical_state(board))
        
    def get_expert_move(self, board, player):
        """Determine the best move according to expert Tic-Tac-Toe strategy."""
        # Convert format if needed
        board_list = board
        if isinstance(board, str):
            board_list = list(board)
        
        # Priority 1: Win if possible
        winning_move = self.find_winning_move(board_list, player)
        if winning_move is not None:
            return winning_move
        
        # Priority 2: Block opponent's win
        opponent = 'O' if player == 'X' else 'X'
        blocking_move = self.find_winning_move(board_list, opponent)
        if blocking_move is not None:
            return blocking_move
        
        # Priority 3: Take center if available
        if board_list[4] == " ":
            return 4
        
        # Priority 4: Take corner if opponent has center
        if board_list[4] == opponent:
            corners = [0, 2, 6, 8]
            available_corners = [corner for corner in corners if board_list[corner] == " "]
            if available_corners:
                return random.choice(available_corners)
        
        # Priority 5: Create a fork
        fork_move = self.find_fork_move(board_list, player)
        if fork_move is not None:
            return fork_move
        
        # Priority 6: Block opponent's fork
        opponent_fork = self.find_fork_move(board_list, opponent)
        if opponent_fork is not None:
            return opponent_fork
        
        # Priority 7: Take any corner
        corners = [0, 2, 6, 8]
        available_corners = [corner for corner in corners if board_list[corner] == " "]
        if available_corners:
            return random.choice(available_corners)
        
        # Priority 8: Take any edge
        edges = [1, 3, 5, 7]
        available_edges = [edge for edge in edges if board_list[edge] == " "]
        if available_edges:
            return random.choice(available_edges)
        
        # If nothing else, take any available move
        valid_moves = self.get_valid_moves(board_list)
        if valid_moves:
            return random.choice(valid_moves)
        
        return None
    
    def find_winning_move(self, board, player):
        """Find an immediate winning move if it exists."""
        for i in range(9):
            if board[i] == " ":
                # Try this move
                board_copy = board.copy()
                board_copy[i] = player
                
                # Check if it's a win
                if self.check_winner(board_copy) == player:
                    return i
        return None
        
    def find_fork_move(self, board, player):
        """Find a move that creates two winning ways (a fork)."""
        valid_moves = self.get_valid_moves(board)
        for move in valid_moves:
            # Try this move
            board_copy = board.copy()
            board_copy[move] = player
            
            # Count how many winning ways this creates
            winning_ways = 0
            for test_move in self.get_valid_moves(board_copy):
                test_board = board_copy.copy()
                test_board[test_move] = player
                if self.check_winner(test_board) == player:
                    winning_ways += 1
            
            # If it creates 2+ winning ways, it's a fork
            if winning_ways >= 2:
                return move
        
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
            if (episode + 1) % window_size == 0 and episode > 0:
                win_rate = wins / window_size
                self.training_stats['wins'].append(win_rate)
                loss_rate = losses / window_size
                self.training_stats['losses'].append(loss_rate)
                draw_rate = draws / window_size
                self.training_stats['draws'].append(draw_rate)
                            
                # Print current stats
                print(f"Episode: {episode + 1}/{episodes}, Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}, Epsilon: {self.epsilon:.3f}")
                
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
        wins, losses, draws = 0, 0, 0
        stats_window = 1000  # Number of episodes between stats updates

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
                    game_history.append((state, action, reward, next_state, done, "O"))
                
                # Check if game is over and track result
                if done:
                    winner = self.check_winner(board)
                    if winner == "O":
                        wins += 1
                    elif winner == "X":
                        losses += 1
                    elif winner == "draw":
                        draws += 1
                    break
                
                current_player = "O" if current_player == "X" else "X"
            
            # Learning phase - only update AI's Q-table
            for state, action, reward, next_state, done, player in game_history:
                self.update_q_value(state, action, reward, next_state, self.q_table)
                self.add_to_replay_buffer((state, action, reward, next_state, done, player))
            
            if episode % 10 == 0:
                self.learn_from_replay_buffer()

            if (episode + 1) % stats_window == 0 and episode > 0:
                win_rate = wins / stats_window
                loss_rate = losses / stats_window
                draw_rate = draws / stats_window
    
                # Store stats
                self.training_stats['wins'].append(win_rate)
                self.training_stats['losses'].append(loss_rate)
                self.training_stats['draws'].append(draw_rate)
                            
                # Print current stats
                print(f"Episode: {episode + 1}/{episodes}, Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}, Epsilon: {self.epsilon:.3f}")

                # Reset counters
                wins, losses, draws = 0, 0, 0
    
    def train_against_minimax(self, episodes=5000, depth_limit=1):
        """Train against a minimax opponent with limited depth."""
        wins, losses, draws = 0, 0, 0
        stats_window = 1000  # Number of episodes between stats updates
        
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
                    move_2d = self.minimax_ai.best_move(board_2d, depth_limit)
                    
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
                    game_history.append((state, action, reward, next_state, done, "O"))
                
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
            
            # Learning phase - only update AI's Q-table
            for state, action, reward, next_state, done, player in game_history:
                self.update_q_value(state, action, reward, next_state, self.q_table)
                self.add_to_replay_buffer((state, action, reward, next_state, done, player))
            
            if episode % 10 == 0:
                self.learn_from_replay_buffer()

            if (episode + 1) % stats_window == 0 and episode > 0:
                win_rate = wins / stats_window
                loss_rate = losses / stats_window
                draw_rate = draws / stats_window
    
                # Store stats
                self.training_stats['wins'].append(win_rate)
                self.training_stats['losses'].append(loss_rate)
                self.training_stats['draws'].append(draw_rate)
                            
                # Print current stats
                print(f"Episode: {episode + 1}/{episodes}, Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}, Epsilon: {self.epsilon:.3f}")

                # Reset counters
                wins, losses, draws = 0, 0, 0

    def play_move(self, board):
        """Make a move during an actual game with enhanced move selection."""
        board_1d = self.convert_2d_to_1d(board) if isinstance(board, list) and len(board) <= 3 else board
        
        # First check for immediate winning move
        winning_move = self.find_winning_move(board_1d, self.player)
        if winning_move is not None:
            return winning_move
        
        # Then check for immediate blocking move
        opponent = 'X' if self.player == 'O' else 'O'
        blocking_move = self.find_winning_move(board_1d, opponent)
        if blocking_move is not None:
            return blocking_move
        
        # Use fork strategy if possible
        fork_move = self.find_fork_move(board_1d, self.player)
        if fork_move is not None:
            return fork_move
        
        # Block opponent's fork
        opponent_fork = self.find_fork_move(board_1d, opponent)
        if opponent_fork is not None:
            return opponent_fork
        
        # Otherwise use Q-table with temperature-based selection
        state_key = self.get_state_key(board_1d)
        valid_moves = self.get_valid_moves(board_1d)
        
        if state_key in self.q_table:
            q_values = np.array(self.q_table[state_key])
            
            # Mask invalid moves
            masked_q_values = np.full(9, -float('inf'))
            for move in valid_moves:
                masked_q_values[move] = q_values[move]
            
            # Temperature-based selection
            temperature = 0.5
            exp_values = np.exp(masked_q_values/temperature)
            probs = exp_values / np.sum(exp_values)
            
            return np.random.choice(9, p=probs)
        else:
            # If state not in Q-table, use heuristics
            if board_1d[4] == " ":  # Take center if available
                return 4
            
            # Take corner if available
            for corner in [0, 2, 6, 8]:
                if board_1d[corner] == " ":
                    return corner
            
            return random.choice(valid_moves)
    
    def best_move(self, board_2d):
        """Return the best move as (row, col) coordinates."""
        board_1d = self.convert_2d_to_1d(board_2d)
        action = self.play_move(board_1d)
        return self.convert_index_to_coordinates(action)
    
    def save_q_table(self, filename=None):
        try:
            model_path = get_model_path("q_learning") if filename is None else filename
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump((self.q_table, self.opponent_q_table, self.training_stats), f)
            print(f"Q-table saved to {model_path}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load_q_table(self, filename=None):
        try:
            model_path = get_model_path("q_learning") if filename is None else filename
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                    if len(data) == 2:
                        self.q_table, self.opponent_q_table = data
                    else:
                        self.q_table, self.opponent_q_table, self.training_stats = data
                    print(f"Q-table loaded from {model_path}")
                    return True  # Successfully loaded
            else:
                print(f"No pre-trained Q-table found at {model_path}. Starting fresh.")
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
        """Enhanced curriculum learning for Q-learning with expert pre-training."""
        # Start with expert pre-training
        print("Starting pre-training with expert moves...")
        self.pretrain_on_expert_moves(num_examples=5000)
        
        total_episodes = episodes
        print("Stage 1: Training against random opponent...")
        self.train_against_random(int(total_episodes * 0.35))
        
        print("Stage 2: Training against easy minimax...")
        self.train_against_minimax(int(total_episodes * 0.15), depth_limit=1)
        
        print("Stage 3: Training against harder minimax...")
        self.train_against_minimax(int(total_episodes * 0.25), depth_limit=3)
        
        print("Stage 4: Training with full self-play...")
        self.train(int(total_episodes * 0.25))
        
        print("Curriculum training complete!")

    def initialize_with_heuristics(self):
        """Initialize Q-table with heuristic values."""
        print("Initializing Q-table with heuristics...")
        # Values for empty board
        empty_board = tuple([" "] * 9)
        
        # Prefer center
        if empty_board not in self.q_table:
            self.q_table[empty_board] = [0.0] * 9
        self.q_table[empty_board][4] = 0.3
        
        # Then corners
        for corner in [0, 2, 6, 8]:
            if empty_board not in self.q_table:
                self.q_table[empty_board] = [0.0] * 9
            self.q_table[empty_board][corner] = 0.2
        
        # Sides are less valuable
        for side in [1, 3, 5, 7]:
            if empty_board not in self.q_table:
                self.q_table[empty_board] = [0.0] * 9
            self.q_table[empty_board][side] = 0.1
        
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
                    if state not in q_table:
                        q_table[state] = [0.0] * 9
                    q_table[state][empty_pos] = 0.9  # High value for completing a win
                    
                    # Create a board with two opponent marks in a row and initialize high Q-value for blocking move
                    board = [" "] * 9
                    for pos in other_pos:
                        board[pos] = opponent
                    
                    state = self.get_canonical_state(board)
                    if state not in q_table:
                        q_table[state] = [0.0] * 9
                    q_table[state][empty_pos] = 0.7  # High value for blocking opponent's win

    def choose_action_ucb(self, board, q_table, game_mode=False):
        """Choose action using UCB exploration strategy."""
        state = self.get_state(board)
        valid_moves = self.get_valid_moves(board)
        
        if not valid_moves:
            return None
        
        # In actual gameplay, use greedy strategy
        if game_mode:
            state_key = self.get_state_key(state)
            q_values = []
            for move in valid_moves:
                if state_key in q_table:
                    q_values.append(q_table[state_key][move])
                else:
                    q_values.append(0)
            max_q = max(q_values)
            best_moves = [valid_moves[i] for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_moves)
        
        # During training, use UCB
        total_visits = sum(self.visit_counts.get((state, move), 0) + 1 for move in valid_moves)
        ucb_values = []
        
        state_key = self.get_state_key(state)
        for move in valid_moves:
            if state_key in q_table:
                q_value = q_table[state_key][move]
            else:
                q_value = 0
            visit_count = self.visit_counts.get((state, move), 0) + 1
            exploration = 1.0 * math.sqrt(math.log(total_visits + 1) / visit_count)
            ucb_values.append(q_value + exploration)
        
        best_move = valid_moves[ucb_values.index(max(ucb_values))]
        return best_move

    def visualize_q_values(self, board):
        """Visualize Q-values for current board state."""
        state = self.get_state(board)
        state_key = self.get_state_key(state) if isinstance(state, tuple) else self.get_state_key(board)
        results = []
        
        for i in range(9):
            if board[i] == " ":
                # FIX: Access q_value correctly given the q_table structure
                q_value = self.q_table.get(state_key, [0]*9)[i] if state_key in self.q_table else 0
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

    def plot_training_progress(self):
        """Plot training progress over time with Seaborn visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            
            if not self.training_stats['wins']:
                print("No training statistics available.")
                return
            
            # Set seaborn style and context
            sns.set_theme(style="darkgrid")
            sns.set_context("notebook", font_scale=1.2)
            
            # Create a pandas DataFrame from training stats
            episodes = np.array(range(len(self.training_stats['wins']))) * 1000
            df = pd.DataFrame({
                'Episode': episodes,
                'Win Rate': self.training_stats['wins'],
                'Loss Rate': self.training_stats['losses'],
                'Draw Rate': self.training_stats['draws']
            })
            
            # Create figure with larger size
            plt.figure(figsize=(12, 7))
            
            # Plot with seaborn
            ax = plt.gca()
            
            # Plot each metric separately
            sns.lineplot(x='Episode', y='Win Rate', data=df,
                         label='Wins', color='green', ax=ax)
            sns.lineplot(x='Episode', y='Loss Rate', data=df,
                         label='Losses', color='red', ax=ax)
            sns.lineplot(x='Episode', y='Draw Rate', data=df, 
                         label='Draws', color='blue', ax=ax)
            
            # Enhance the plot
            plt.title('Q-Learning Agent Training Progress', fontsize=16, pad=20)
            plt.xlabel('Training Episodes', fontsize=14)
            plt.ylabel('Rate', fontsize=14)
            
            # Improve legend
            plt.legend(title='Metrics', title_fontsize=13, fontsize=12, 
                     frameon=True, facecolor='white', edgecolor='gray')
            
            # Add annotations for the final rates
            if len(episodes) > 0:
                latest_win = df['Win Rate'].iloc[-1]
                latest_loss = df['Loss Rate'].iloc[-1]
                latest_draw = df['Draw Rate'].iloc[-1]
                
                plt.annotate(f"Final Win Rate: {latest_win:.2f}", 
                           xy=(episodes[-1], latest_win),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=11, color='green')
                           
                plt.annotate(f"Final Loss Rate: {latest_loss:.2f}", 
                           xy=(episodes[-1], latest_loss),
                           xytext=(10, -15), textcoords='offset points', 
                           fontsize=11, color='red')
                           
                plt.annotate(f"Final Draw Rate: {latest_draw:.2f}", 
                           xy=(episodes[-1], latest_draw),
                           xytext=(10, -40), textcoords='offset points',
                           fontsize=11, color='blue')
            
            # Add some padding to the layout
            plt.tight_layout()
            
            # Save with high DPI
            plot_path = get_plot_path("q_learning")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {plot_path}")
            
        except ImportError as e:
            print(f"Required plotting libraries not available: {e}")
            print("Install matplotlib, pandas, and seaborn for visualization.")