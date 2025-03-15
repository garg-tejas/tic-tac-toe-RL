import random
import math
import copy
import time
import os
import pickle
import numpy as np
from utils import get_model_path, get_plot_path

class MCTSNode:
    """Node in the Monte Carlo search tree representing a game state."""
    
    def __init__(self, board, parent=None, move=None, player=None):
        """Initialize a new tree node."""
        self.board = board  # Game state at this node
        self.parent = parent  # Parent node
        self.move = move  # Move that led to this state
        self.player = player  # Player who made the move
        self.children = []  # Child nodes
        self.wins = 0  # Number of wins from simulations that passed through this node
        self.visits = 0  # Number of simulations that passed through this node
        
        # Handle None board case
        self.untried_moves = [] if board is None else self.get_valid_moves(board)
        
        # Add hash for transposition table - handle None case
        self.board_hash = None if board is None else self.hash_board(board)
        
        # Add RAVE statistics
        self.amaf_wins = {}  # All-Moves-As-First statistics
        self.amaf_visits = {}
        # Add virtual loss counter for parallel search
        self.virtual_loss = 0
            
    def hash_board(self, board):
        """Create a hash of the board for transposition table."""
        board_str = ''.join(['.' if cell is None else cell for row in board for cell in row])
        return hash(board_str)
    
    def get_valid_moves(self, board):
        """Return a list of valid move coordinates."""
        if board is None:
            return []  # Return empty list for None board
        return [(r, c) for r in range(3) for c in range(3) if board[r][c] is None]
    
    def is_fully_expanded(self):
        """Check if all possible moves have been expanded."""
        return len(self.untried_moves) == 0
    
    def best_child(self, exploration_weight, use_rave=False, rave_weight=0.1):
        """Select best child using PUCT formula with optional RAVE."""
        if not self.children:
            return None
            
        best_value = float('-inf')
        best_child = None
        
        # Get total visits for normalization
        log_visits = math.log(self.visits + 1)
        
        for child in self.children:
            # Basic UCB1
            exploit = child.wins / (child.visits + 1e-6)
            explore = exploration_weight * math.sqrt(log_visits / (child.visits + 1e-6))
            
            # Add virtual loss for parallel search
            virtual_loss_penalty = 0.1 * child.virtual_loss
            
            # Add RAVE component if enabled
            rave_score = 0
            if use_rave and child.move in self.amaf_visits and self.amaf_visits[child.move] > 0:
                rave_exploit = self.amaf_wins.get(child.move, 0) / self.amaf_visits[child.move]
                beta = self.amaf_visits[child.move] / (self.amaf_visits[child.move] + child.visits + 4 * rave_weight * self.amaf_visits[child.move] * child.visits)
                rave_score = beta * rave_exploit
                
            # Add heuristic bias for important positions
            position_bias = 0
            if isinstance(child.move, tuple) and len(child.move) == 2:
                # Center is valuable in tic-tac-toe
                if child.move == (1, 1):
                    position_bias = 0.1
                # Corners are next most valuable
                elif child.move in [(0,0), (0,2), (2,0), (2,2)]:
                    position_bias = 0.05
            
            value = exploit + explore + rave_score + position_bias - virtual_loss_penalty
            
            if value > best_value:
                best_value = value
                best_child = child
                
        return best_child
    

class MCTSAI:
    """Monte Carlo Tree Search implementation for Tic-Tac-Toe."""
    
    def __init__(self, exploration_weight=1.41, time_limit=0.5, temperature=0.1):
        self.exploration_weight = exploration_weight
        self.time_limit = time_limit
        self.temperature = temperature
        self.player = 'O'  # Default player
        self.node_pool = MCTSNodePool(initial_size=500)
        self.transposition_table = TranspositionTable(max_size=50000)
        
        # Add training statistics like other implementations
        self.training_stats = {'wins': [], 'losses': [], 'draws': []}
        
        # Track trained knowledge
        self.position_values = {}  # Store value estimates for positions
        
        # Try to load pre-trained data
        self.load_model()
    
    def set_player(self, player):
        """Set the player marker (X or O)."""
        self.player = player
        
    def best_move(self, board_2d, time_limit=None, temperature=None):
        """Return the best move within a time limit (seconds)."""
        # Use passed parameters or default instance values
        time_limit = time_limit if time_limit is not None else self.time_limit
        temperature = temperature if temperature is not None else self.temperature
        
        # Create root node
        root = self.node_pool.get_node(board_2d, player=self.player)
        start_time = time.time()
        sim_count = 0
        
        # Create a transposition table for this search
        transpositions = {}
        
        # Handle special case - if there's only one valid move
        if len(root.untried_moves) == 1:
            return root.untried_moves[0]
        
        # Use time-based constraint instead of fixed simulation count
        while time.time() - start_time < time_limit:
            # 1. Selection phase - select a promising leaf node
            node = self.select_node(root)
            
            # 2. Check transposition table for known states
            if node.board_hash in transpositions:
                # Get cached information
                cached_stats = transpositions[node.board_hash]
                # Update the node with the cached statistics
                node.visits += cached_stats[0]
                node.wins += cached_stats[1]
                continue
            
            # 3. Add virtual loss to encourage thread divergence in parallel search
            node.virtual_loss += 1
            
            # 4. Expansion phase - expand if not fully expanded
            if not node.is_fully_expanded():
                node = self.expand_node(node)
            
            # 5. Simulation phase - play out a random game from this position
            result, move_history = self.simulate(copy.deepcopy(node.board), node.player)
            
            # 6. Backpropagation phase - update statistics up the tree
            self.backpropagate(node, result, move_history)
            
            # 7. Store in transposition table
            transpositions[node.board_hash] = (node.visits, node.wins)
            
            sim_count += 1
        
        # Choose best move based on temperature parameter
        if not root.children:
            # No children, pick from untried moves
            if root.untried_moves:
                return random.choice(root.untried_moves)
            else:
                # No valid moves, should never happen in a valid game
                return (1, 1)  # Default to center as fallback
        
        if temperature < 0.01:
            # Deterministic selection (exploitation)
            best_child = max(root.children, key=lambda c: c.visits)
        else:
            # Temperature-based selection (exploration/exploitation trade-off)
            visits_count = [child.visits for child in root.children]
            # Apply temperature scaling
            if sum(visits_count) > 0:  # Avoid division by zero
                probabilities = [count**(1/temperature) for count in visits_count]
                total = sum(probabilities)
                if total > 0:  # Avoid division by zero
                    probabilities = [p/total for p in probabilities]
                    chosen_index = random.choices(range(len(root.children)), probabilities)[0]
                    best_child = root.children[chosen_index]
                else:
                    best_child = random.choice(root.children)
            else:
                best_child = random.choice(root.children)
        
        # Save the value of this state to our position_values
        board_hash = root.board_hash
        self.position_values[board_hash] = {
            'visits': root.visits,
            'value': root.wins / max(root.visits, 1)
        }
        
        # Cleanup to prevent memory leaks
        self.node_pool.recycle_subtree(root)
        
        return best_child.move

    def train(self, episodes=10000, early_stopping=True, patience=5):
        """Train MCTS by self-play and record statistics for visualization."""
        # For early stopping
        best_win_rate = 0
        no_improvement_count = 0
        
        # For tracking performance
        window_size = 1000
        wins, losses, draws = 0, 0, 0
        
        for episode in range(episodes):
            board = [[None for _ in range(3)] for _ in range(3)]
            current_player = "X"  # X always starts
            game_history = []
            
            # Play out the game
            while True:
                # Use a shorter time limit during training for speed
                move = self.best_move(board, time_limit=0.1)
                
                # Apply move
                board = self.make_move(copy.deepcopy(board), move, current_player)
                
                # Check for game end
                winner = self.check_winner(board)
                if winner or not self.get_valid_moves(board):
                    # Record outcome
                    if winner == self.player:
                        wins += 1
                    elif winner == "draw":
                        draws += 1
                    else:
                        losses += 1
                    break
                
                # Switch player
                current_player = 'O' if current_player == 'X' else 'X'
            
            # Save statistics periodically
            if (episode + 1) % window_size == 0 and episode > 0:
                win_rate = wins / window_size
                loss_rate = losses / window_size
                draw_rate = draws / window_size
                
                # Store stats
                self.training_stats['wins'].append(win_rate)
                self.training_stats['losses'].append(loss_rate)
                self.training_stats['draws'].append(draw_rate)
                
                # Print current stats
                print(f"Episode: {episode + 1}/{episodes}, Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}")
                
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

        # Save the trained model
        self.save_model()
    
    def train_against_random(self, episodes=5000):
        """Train MCTS against a random opponent."""
        wins, losses, draws = 0, 0, 0
        stats_window = 1000  # Number of episodes between stats updates
        
        for episode in range(episodes):
            board = [[None for _ in range(3)] for _ in range(3)]
            current_player = "X"  # X always starts
            
            # Determine player/opponent markers
            mcts_player = self.player
            opponent_player = 'X' if mcts_player == 'O' else 'O'
            
            while True:
                if current_player == mcts_player:
                    # MCTS player's turn
                    move = self.best_move(board, time_limit=0.1)
                else:
                    # Random player's turn
                    moves = self.get_valid_moves(board)
                    move = random.choice(moves) if moves else None
                
                if move:
                    board = self.make_move(copy.deepcopy(board), move, current_player)
                
                # Check for game end
                winner = self.check_winner(board)
                if winner or not self.get_valid_moves(board):
                    # Record outcome from MCTS perspective
                    if winner == mcts_player:
                        wins += 1
                    elif winner == "draw":
                        draws += 1
                    elif winner:
                        losses += 1
                    else:
                        draws += 1  # Full board, no winner
                    break
                
                # Switch player
                current_player = 'O' if current_player == 'X' else 'X'
            
            # Save statistics periodically
            if (episode + 1) % stats_window == 0 and episode > 0:
                win_rate = wins / stats_window
                loss_rate = losses / stats_window
                draw_rate = draws / stats_window
                
                # Store stats
                self.training_stats['wins'].append(win_rate)
                self.training_stats['losses'].append(loss_rate)
                self.training_stats['draws'].append(draw_rate)
                
                # Print current stats
                print(f"Episode: {episode + 1}/{episodes}, Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}")
                
                # Reset counters
                wins, losses, draws = 0, 0, 0
    
    def train_against_minimax(self, episodes=5000, depth_limit=1):
        """Train against a minimax opponent with limited depth."""
        # Import minimax here to prevent circular imports
        from minimax_ai import MinimaxAI
        minimax_ai = MinimaxAI()
        
        wins, losses, draws = 0, 0, 0
        stats_window = 1000  # Number of episodes between stats updates
        
        for episode in range(episodes):
            board = [[None for _ in range(3)] for _ in range(3)]
            current_player = "X"  # X always starts
            
            # Determine player/opponent markers
            mcts_player = self.player
            opponent_player = 'X' if mcts_player == 'O' else 'O'
            
            # Set minimax player
            minimax_ai.set_player(opponent_player)
            
            while True:
                if current_player == mcts_player:
                    # MCTS player's turn
                    move = self.best_move(board, time_limit=0.1)
                else:
                    # Minimax player's turn
                    move = minimax_ai.best_move(board, depth_limit)
                
                if move:
                    board = self.make_move(copy.deepcopy(board), move, current_player)
                
                # Check for game end
                winner = self.check_winner(board)
                if winner or not self.get_valid_moves(board):
                    # Record outcome from MCTS perspective
                    if winner == mcts_player:
                        wins += 1
                    elif winner == "draw":
                        draws += 1
                    elif winner:
                        losses += 1
                    else:
                        draws += 1  # Full board, no winner
                    break
                
                # Switch player
                current_player = 'O' if current_player == 'X' else 'X'
            
            # Save statistics periodically
            if (episode + 1) % stats_window == 0 and episode > 0:
                win_rate = wins / stats_window
                loss_rate = losses / stats_window
                draw_rate = draws / stats_window
                
                # Store stats
                self.training_stats['wins'].append(win_rate)
                self.training_stats['losses'].append(loss_rate)
                self.training_stats['draws'].append(draw_rate)
                
                # Print current stats
                print(f"Episode: {episode + 1}/{episodes}, Win Rate: {win_rate:.2f}, Loss Rate: {loss_rate:.2f}, Draw Rate: {draw_rate:.2f}")
                
                # Reset counters
                wins, losses, draws = 0, 0, 0
    
    def train_curriculum(self, episodes=50000):
        """Train using curriculum learning - increasing difficulty."""
        total_episodes = episodes
        # First stage: train against random opponent
        print("Stage 1: Training against random opponent...")
        self.train_against_random(int(total_episodes * 0.35))
        
        # Second stage: train against easy minimax
        print("Stage 2: Training against easy minimax...")
        self.train_against_minimax(int(total_episodes * 0.15), depth_limit=1)
        
        # Third stage: train against harder minimax
        print("Stage 3: Training against harder minimax...")
        self.train_against_minimax(int(total_episodes * 0.25), depth_limit=3)
        
        # Fourth stage: full self-play
        print("Stage 4: Training with full self-play...")
        self.train(int(total_episodes * 0.25))
        
        print("Curriculum training complete!")
    
    def save_model(self, filename=None):
        """Save MCTS statistics and training data."""
        model_path = get_model_path("mcts") if filename is None else filename
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        data = {
            'position_values': self.position_values,
            'training_stats': self.training_stats,
            'params': {
                'exploration_weight': self.exploration_weight,
                'time_limit': self.time_limit,
                'temperature': self.temperature
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"MCTS model saved to {model_path}")
    
    def load_model(self, filename=None):
        """Load MCTS statistics and training data."""
        try:
            model_path = get_model_path("mcts") if filename is None else filename
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.position_values = data.get('position_values', {})
                self.training_stats = data.get('training_stats', {'wins': [], 'losses': [], 'draws': []})
                
                # Optionally load parameters
                params = data.get('params', {})
                self.exploration_weight = params.get('exploration_weight', self.exploration_weight)
                self.time_limit = params.get('time_limit', self.time_limit)
                self.temperature = params.get('temperature', self.temperature)
                
                print(f"MCTS model loaded from {model_path}")
                return True
            else:
                print(f"No MCTS model found at {model_path}")
                return False
        except Exception as e:
            print(f"Error loading MCTS model: {e}")
            return False
    
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
            
            # Plot each metric with a different style
            sns.lineplot(x='Episode', y='Win Rate', data=df,
                         label='Wins', color='green', ax=ax)
            sns.lineplot(x='Episode', y='Loss Rate', data=df,
                         label='Losses', color='red', ax=ax)
            sns.lineplot(x='Episode', y='Draw Rate', data=df,
                         label='Draws', color='blue', ax=ax)
            
            try:
                # Add smoothed trend lines if we have enough data points
                if len(episodes) >= 4:
                    sns.regplot(x='Episode', y='Win Rate', data=df, scatter=False, 
                              lowess=True, line_kws={'color': 'green', 'alpha': 0.7, 'lw': 2}, ax=ax)
                    sns.regplot(x='Episode', y='Loss Rate', data=df, scatter=False, 
                              lowess=True, line_kws={'color': 'red', 'alpha': 0.7, 'lw': 2}, ax=ax)
                    sns.regplot(x='Episode', y='Draw Rate', data=df, scatter=False, 
                              lowess=True, line_kws={'color': 'blue', 'alpha': 0.7, 'lw': 2}, ax=ax)
            except:
                pass  # Skip trend lines if statsmodels not available
            
            # Enhance the plot
            plt.title('MCTS Agent Training Progress', fontsize=16, pad=20)
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
            plot_path = get_plot_path("mcts")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {plot_path}")
            
        except Exception as e:
            print(f"Could not plot training progress: {e}")
    
    def select_node(self, node):
        """Select a leaf node using UCB1 formula with RAVE."""
        while not node.is_fully_expanded() or node.children:
            if not node.is_fully_expanded():
                return node
            node = node.best_child(
                self.exploration_weight,
                use_rave=True,  # Enable RAVE consistently
                rave_weight=0.1
            )
        return node
    
    def expand_node(self, node):
        """Add a child node with an unexplored move."""
        move = node.untried_moves.pop()
        new_board = self.make_move(copy.deepcopy(node.board), move, node.player)
        next_player = 'X' if node.player == 'O' else 'O'
        
        # Use node pool instead of direct creation
        child_node = self.node_pool.get_node(
            new_board, 
            parent=node, 
            move=move, 
            player=next_player
        )
        
        node.children.append(child_node)
        return child_node
    
    def simulate(self, board, player):
        """Run a simulation with better heuristics than random play."""
        current_player = player
        board_copy = copy.deepcopy(board)
        move_history = []  # Track moves for RAVE
        
        while self.check_winner(board_copy) is None and self.get_valid_moves(board_copy):
            # Check for winning move
            winning_move = self.find_winning_move(board_copy, current_player)
            if winning_move:
                move = winning_move
            # Check for blocking move (prevent opponent win)
            else:
                opponent = 'O' if current_player == 'X' else 'X'
                blocking_move = self.find_winning_move(board_copy, opponent)
                if blocking_move:
                    move = blocking_move
                # Otherwise use weighted random with some domain knowledge
                else:
                    moves = self.get_valid_moves(board_copy)
                    
                    # Prefer center, then corners, then edges with weighted selection
                    weights = []
                    for r, c in moves:
                        if (r, c) == (1, 1):  # Center
                            weights.append(3.0)
                        elif (r, c) in [(0,0), (0,2), (2,0), (2,2)]:  # Corners
                            weights.append(2.0)
                        else:  # Edges
                            weights.append(1.0)
                    
                    total_weight = sum(weights)
                    selection = random.uniform(0, total_weight)
                    current_weight = 0
                    for i, (move_candidate, weight) in enumerate(zip(moves, weights)):
                        current_weight += weight
                        if current_weight >= selection:
                            move = move_candidate
                            break
                    else:
                        move = random.choice(moves)  # Fallback
            
            # Record move for RAVE statistics
            move_history.append((current_player, move))
            
            # Make the move
            board_copy = self.make_move(board_copy, move, current_player)
            current_player = 'O' if current_player == 'X' else 'X'
        
        result = self.check_winner(board_copy)
        return result, move_history
    
    def find_winning_move(self, board, player):
        """Find an immediate winning move if it exists."""
        for r, c in self.get_valid_moves(board):
            board_copy = copy.deepcopy(board)
            board_copy[r][c] = player
            if self.check_winner(board_copy) == player:
                return (r, c)
        return None
    
    def backpropagate(self, node, result, move_history=None):
        """Update statistics on the path back to the root."""
        # Standard backpropagation
        while node:
            node.visits += 1
            
            # Update win count for node's player perspective
            if result == node.player:
                node.wins += 1
            elif result == "draw":
                node.wins += 0.5
                
            # Decrease virtual loss (used for parallel search)
            if hasattr(node, 'virtual_loss') and node.virtual_loss > 0:
                node.virtual_loss -= 1
                
            # Update RAVE statistics if move history is provided
            if move_history and node.parent:
                player = node.parent.player  # Get player who made the move to this node
                for move_player, move in move_history:
                    if move_player == player:
                        # Initialize if not existing
                        if move not in node.parent.amaf_visits:
                            node.parent.amaf_visits[move] = 0
                            node.parent.amaf_wins[move] = 0
                        
                        # Update AMAF statistics
                        node.parent.amaf_visits[move] += 1
                        if result == player:
                            node.parent.amaf_wins[move] += 1
                        elif result == "draw":
                            node.parent.amaf_wins[move] += 0.5
            
            node = node.parent
    
    def check_winner(self, board):
        """Check if there is a winner or a draw on a 2D board."""
        # Check rows and columns
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not None:
                return board[i][0]  # Row win
            if board[0][i] == board[1][i] == board[2][i] and board[0][i] is not None:
                return board[0][i]  # Column win
        
        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
            return board[0][0]  # Diagonal win
        if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
            return board[0][2]  # Diagonal win
                
        # Check for draw/ongoing
        if any(None in row for row in board):
            return None  # Game still ongoing
        return "draw"
    
    def get_valid_moves(self, board):
        """Return a list of valid move coordinates."""
        return [(r, c) for r in range(3) for c in range(3) if board[r][c] is None]
    
    def make_move(self, board, move, player):
        """Apply a move to the board and return the new board state."""
        board[move[0]][move[1]] = player
        return board
    
    def convert_2d_to_1d(self, board_2d):
        """Convert 2D board to 1D format."""
        board_1d = [" "] * 9
        for i in range(3):
            for j in range(3):
                cell = board_2d[i][j]
                if cell is not None:
                    board_1d[i*3 + j] = cell
        return board_1d
    
    def convert_1d_to_2d(self, board_1d):
        """Convert 1D board to 2D format."""
        board_2d = [[None for _ in range(3)] for _ in range(3)]
        for i in range(9):
            row, col = i // 3, i % 3
            board_2d[row][col] = board_1d[i] if board_1d[i] != " " else None
        return board_2d
    

class MCTSNodePool:
    """Pool of reusable MCTS nodes to reduce memory allocation."""
    
    def __init__(self, initial_size=1000):
        # Create empty nodes without trying to compute valid moves yet
        self.free_nodes = []
        for _ in range(initial_size):
            node = MCTSNode(None)
            node.untried_moves = []  # Explicitly set to empty list
            self.free_nodes.append(node)
        
    def get_node(self, board, parent=None, move=None, player=None):
        """Get a node from the pool or create a new one."""
        if not self.free_nodes:
            # Expand pool if needed
            self.free_nodes.extend([MCTSNode(None) for _ in range(1000)])
            
        node = self.free_nodes.pop()
        # Reset and initialize the node
        node.board = board
        node.parent = parent
        node.move = move
        node.player = player
        node.children = []
        node.wins = 0
        node.visits = 0
        node.untried_moves = node.get_valid_moves(board)
        node.board_hash = node.hash_board(board) if board else None
        node.amaf_wins = {}
        node.amaf_visits = {}
        node.virtual_loss = 0
        
        return node
        
    def recycle_node(self, node):
        """Return a node to the pool."""
        # Clear references to prevent memory leaks
        node.children = []
        node.parent = None
        node.board = None
        self.free_nodes.append(node)
        
    def recycle_subtree(self, node):
        """Recycle a node and all its children."""
        for child in node.children:
            self.recycle_subtree(child)
        self.recycle_node(node)


class TranspositionTable:
    """Table storing previously seen board positions."""
    
    def __init__(self, max_size=100000):
        self.table = {}
        self.max_size = max_size
        
    def get(self, board_hash):
        """Get stored statistics for a board position."""
        return self.table.get(board_hash)
        
    def store(self, board_hash, visits, wins):
        """Store statistics for a board position."""
        if len(self.table) >= self.max_size:
            # Simple LRU: remove a random entry when table is full
            # In a real implementation, you'd use a proper LRU strategy
            self.table.pop(next(iter(self.table)))
            
        self.table[board_hash] = (visits, wins)