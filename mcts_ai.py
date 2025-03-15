import random
import math
import copy
import time
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_model_path, get_plot_path


class PolicyNetwork(nn.Module):
    """Neural network for move prediction and position evaluation."""
    
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Input: 3 channels (player pieces, opponent pieces, empty spaces)
        # for 3x3 board
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.policy_head = nn.Linear(128, 9)  # 9 possible moves
        self.value_head = nn.Linear(128, 1)   # Board evaluation
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 3 * 3)
        x = torch.relu(self.fc1(x))
        policy = torch.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value
    

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

        # Add GPU device detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        # Add neural network for policy and value prediction
        self.policy_net = PolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Track trained knowledge
        self.position_values = {}  # Store value estimates for positions
        
        # Try to load pre-trained data
        self.load_model()
    
    def set_player(self, player):
        """Set the player marker (X or O)."""
        self.player = player
        
    def best_move(self, board_2d, time_limit=None, temperature=None):
        """Return the best move within a time limit (seconds) using GPU batch processing."""
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
        
        # For batch processing
        batch_size = 16  # Adjust based on your GPU memory
        pending_nodes = []
        pending_boards = []
        pending_players = []
        
        # Use time-based constraint instead of fixed simulation count
        while time.time() - start_time < time_limit:
            # Collect nodes for batch processing
            while len(pending_nodes) < batch_size and time.time() - start_time < time_limit * 0.9:
                # Selection phase - select a promising leaf node
                node = self.select_node(root)
                
                # Check transposition table for known states
                if node.board_hash in transpositions:
                    # Get cached information
                    cached_stats = transpositions[node.board_hash]
                    # Update the node with the cached statistics
                    node.visits += cached_stats[0]
                    node.wins += cached_stats[1]
                    continue
                
                # Add virtual loss to encourage thread divergence in parallel search
                node.virtual_loss += 1
                
                # Expansion phase - expand if not fully expanded
                if not node.is_fully_expanded():
                    node = self.expand_node(node)
                
                # Add to pending batch
                pending_nodes.append(node)
                pending_boards.append(copy.deepcopy(node.board))
                pending_players.append(node.player)
                
            # Process batch if we have nodes or time is almost up
            if pending_nodes and (len(pending_nodes) >= batch_size or 
                                time.time() - start_time >= time_limit * 0.9):
                # Perform batch simulation
                results, move_histories = self.batch_simulate(pending_boards, pending_players)
                
                # Backpropagate results
                for node, result, move_history in zip(pending_nodes, results, move_histories):
                    self.backpropagate(node, result, move_history)
                    # Store in transposition table
                    transpositions[node.board_hash] = (node.visits, node.wins)
                
                sim_count += len(pending_nodes)
                
                # Clear pending lists
                pending_nodes = []
                pending_boards = []
                pending_players = []
        
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
    
    def pretrain_on_expert_moves(self, num_examples=10000, epochs=10):
        """Pre-train the neural network on expert Tic-Tac-Toe moves."""
        print("Pre-training on expert moves...")
        
        # Containers for training data
        training_states = []
        training_policies = []
        training_values = []
        
        # Generate expert data
        for _ in range(num_examples):
            # Create a random board with 1-5 moves already played
            board = [[None for _ in range(3)] for _ in range(3)]
            moves_made = random.randint(1, 5)
            current_player = 'X'  # Start with X
            
            # Make random moves to reach a mid-game position
            for _ in range(moves_made):
                valid_moves = self.get_valid_moves(board)
                if not valid_moves:
                    break
                move = random.choice(valid_moves)
                board = self.make_move(copy.deepcopy(board), move, current_player)
                current_player = 'O' if current_player == 'X' else 'X'
                
                # Stop if game is over
                if self.check_winner(board):
                    break
            
            # If game is already over, skip this example
            winner = self.check_winner(board)
            if winner:
                continue
                
            # Now find the "expert move" for this position
            best_move = self.get_expert_move(board, current_player)
            if best_move:
                # Create one-hot policy vector
                policy = torch.zeros(9, device=self.device)
                move_index = best_move[0] * 3 + best_move[1]
                policy[move_index] = 1.0
                
                # Estimate position value (simplified)
                value = 0.8  # Assume expert moves lead to good outcomes
                
                # Store the example
                training_states.append(board)
                training_policies.append(policy)
                training_values.append(value)
        
        # Train for multiple epochs on this data
        print(f"Generated {len(training_states)} expert examples")
        for epoch in range(epochs):
            # Shuffle the data
            indices = list(range(len(training_states)))
            random.shuffle(indices)
            shuffled_states = [training_states[i] for i in indices]
            shuffled_policies = [training_policies[i] for i in indices]
            shuffled_values = [training_values[i] for i in indices]
            
            # Train in batches
            batch_size = 256
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(shuffled_states), batch_size):
                batch_states = shuffled_states[i:i+batch_size]
                batch_policies = shuffled_policies[i:i+batch_size]
                batch_values = shuffled_values[i:i+batch_size]
                
                loss = self.train_policy_network(batch_states, batch_policies, batch_values)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / max(1, num_batches)
            print(f"Pre-training epoch {epoch+1}/{epochs}, Avg loss: {avg_loss:.4f}")
        
        print("Pre-training complete")
    
    def get_expert_move(self, board, player):
        """Determine the best move according to expert Tic-Tac-Toe strategy."""
        # Priority 1: Win if possible
        winning_move = self.find_winning_move(board, player)
        if winning_move:
            return winning_move
        
        # Priority 2: Block opponent's win
        opponent = 'O' if player == 'X' else 'X'
        blocking_move = self.find_winning_move(board, opponent)
        if blocking_move:
            return blocking_move
        
        # Priority 3: Take center if available
        if board[1][1] is None:
            return (1, 1)
        
        # Priority 4: Take corner if opponent has center
        if board[1][1] == opponent:
            corners = [(0,0), (0,2), (2,0), (2,2)]
            available_corners = [corner for corner in corners if board[corner[0]][corner[1]] is None]
            if available_corners:
                return random.choice(available_corners)
        
        # Priority 5: Create a fork (two potential winning ways)
        fork_move = self.find_fork_move(board, player)
        if fork_move:
            return fork_move
        
        # Priority 6: Block opponent's fork
        opponent_fork = self.find_fork_move(board, opponent)
        if opponent_fork:
            return opponent_fork
        
        # Priority 7: Take any corner
        corners = [(0,0), (0,2), (2,0), (2,2)]
        available_corners = [corner for corner in corners if board[corner[0]][corner[1]] is None]
        if available_corners:
            return random.choice(available_corners)
        
        # Priority 8: Take any edge
        edges = [(0,1), (1,0), (1,2), (2,1)]
        available_edges = [edge for edge in edges if board[edge[0]][edge[1]] is None]
        if available_edges:
            return random.choice(available_edges)
        
        # If nothing else, take any available move
        valid_moves = self.get_valid_moves(board)
        if valid_moves:
            return random.choice(valid_moves)
        
        return None

    def find_fork_move(self, board, player):
        """Find move that creates two winning ways (a fork)."""
        valid_moves = self.get_valid_moves(board)
        for move in valid_moves:
            # Try this move
            board_copy = copy.deepcopy(board)
            board_copy[move[0]][move[1]] = player
            
            # Count how many winning ways this creates
            winning_ways = 0
            for test_move in self.get_valid_moves(board_copy):
                test_board = copy.deepcopy(board_copy)
                test_board[test_move[0]][test_move[1]] = player
                if self.check_winner(test_board) == player:
                    winning_ways += 1
            
            # If it creates 2+ winning ways, it's a fork
            if winning_ways >= 2:
                return move
        
        return None

    def train(self, episodes=10000, early_stopping=True, patience=5):
        """Train MCTS by self-play and record statistics for visualization."""
        # For early stopping
        best_win_rate = 0
        no_improvement_count = 0
        
        # For tracking performance
        window_size = 1000
        wins, losses, draws = 0, 0, 0
        
        # For neural network training
        training_states = []
        training_policies = []
        training_values = []
        
        for episode in range(episodes):
            board = [[None for _ in range(3)] for _ in range(3)]
            current_player = "X"  # X always starts
            game_history = []
            
            # Play out the game
            while True:
                # Store current state for training
                game_history.append((copy.deepcopy(board), current_player))
                
                # Use a shorter time limit during training for speed
                move = self.best_move(board, time_limit=0.1)
                
                # Apply move
                board = self.make_move(copy.deepcopy(board), move, current_player)
                
                # Store move information for policy training
                move_index = move[0] * 3 + move[1]
                policy = torch.zeros(9, device=self.device)
                policy[move_index] = 1.0
                
                # Store the move that was taken from this state
                game_history[-1] = (game_history[-1][0], game_history[-1][1], move_index, policy)
                
                # Check for game end
                winner = self.check_winner(board)
                if winner or not self.get_valid_moves(board):
                    # Record outcome
                    if winner == self.player:
                        wins += 1
                        game_value = 1.0
                    elif winner == "draw":
                        draws += 1
                        game_value = 0.5
                    else:
                        losses += 1
                        game_value = 0.0
                    break
                
                # Switch player
                current_player = 'O' if current_player == 'X' else 'X'
            
            # Process game history for neural network training
            for state, player, move_idx, policy in game_history:
                # Adjust value based on player perspective
                player_value = game_value if player == self.player else 1.0 - game_value
                
                # Store training data
                training_states.append(state)
                training_policies.append(policy)
                training_values.append(player_value)
            
            # Train neural network periodically
            if len(training_states) >= 256 or (episode + 1) % 100 == 0:
                if training_states:
                    # Use the latest 2048 examples to avoid memory issues
                    batch_size = min(2048, len(training_states))
                    loss = self.train_policy_network(
                        training_states[-batch_size:],
                        training_policies[-batch_size:],
                        training_values[-batch_size:]
                    )
                    # print(f"Episode {episode+1}, NN Training loss: {loss:.4f}")
                    
                    # Clear training data after using it
                    training_states = []
                    training_policies = []
                    training_values = []
            
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
                        # Save best model
                        self.save_model()
                    else:
                        no_improvement_count += 1
                    
                    if no_improvement_count >= patience:
                        print(f"Early stopping at episode {episode}, no improvement for {patience} windows")
                        break
                
                # Reset counters
                wins, losses, draws = 0, 0, 0
    
        # Save the final model
        self.save_model()
    
    def train_against_random(self, episodes=5000):
        """Train MCTS against a random opponent."""
        wins, losses, draws = 0, 0, 0
        stats_window = 1000  # Number of episodes between stats updates
        
        # For neural network training
        training_states = []
        training_policies = []
        training_values = []
        
        for episode in range(episodes):
            board = [[None for _ in range(3)] for _ in range(3)]
            current_player = "X"  # X always starts
            game_history = []
            
            # Determine player/opponent markers
            mcts_player = random.choice(['X', 'O'])
            opponent_player = 'X' if mcts_player == 'O' else 'O'
            
            while True:
                # Store state information when it's MCTS player's turn
                if current_player == mcts_player:
                    game_history.append((copy.deepcopy(board), current_player))
                
                if current_player == mcts_player:
                    # MCTS player's turn
                    move = self.best_move(board, time_limit=0.1)
                    
                    # Store move information for policy training
                    if game_history:
                        move_index = move[0] * 3 + move[1]
                        policy = torch.zeros(9, device=self.device)
                        policy[move_index] = 1.0
                        
                        # Update the last stored state with move information
                        game_history[-1] = (game_history[-1][0], game_history[-1][1], move_index, policy)
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
                        game_value = 1.0
                    elif winner == "draw":
                        draws += 1
                        game_value = 0.5
                    else:
                        losses += 1
                        game_value = 0.0
                    break
                
                # Switch player
                current_player = 'O' if current_player == 'X' else 'X'
            
            # Process game history for neural network training
            for state, player, move_idx, policy in game_history:
                # Only store MCTS player's moves for training
                if player == mcts_player:
                    # Store training data
                    training_states.append(state)
                    training_policies.append(policy)
                    training_values.append(game_value)
            
            # Train neural network periodically
            if len(training_states) >= 256 or (episode + 1) % 100 == 0:
                if training_states:
                    # Use the latest 2048 examples to avoid memory issues
                    batch_size = min(2048, len(training_states))
                    loss = self.train_policy_network(
                        training_states[-batch_size:],
                        training_policies[-batch_size:],
                        training_values[-batch_size:]
                    )
                    # print(f"Episode {episode+1}, NN Training loss: {loss:.4f}")
                    
                    # Clear training data after using it
                    training_states = []
                    training_policies = []
                    training_values = []
            
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
                
                # Save intermediate model
                if episode % (stats_window * 5) == 0:
                    self.save_model()
                
                # Reset counters
                wins, losses, draws = 0, 0, 0
    
    def train_against_minimax(self, episodes=5000, depth_limit=1):
        """Train against a minimax opponent with limited depth."""
        # Import minimax here to prevent circular imports
        from minimax_ai import MinimaxAI
        minimax_ai = MinimaxAI()
        
        wins, losses, draws = 0, 0, 0
        stats_window = 1000  # Number of episodes between stats updates
        
        # For neural network training
        training_states = []
        training_policies = []
        training_values = []
        
        for episode in range(episodes):
            board = [[None for _ in range(3)] for _ in range(3)]
            current_player = "X"  # X always starts
            game_history = []
            
            # Determine player/opponent markers
            mcts_player = random.choice(['X', 'O'])
            opponent_player = 'X' if mcts_player == 'O' else 'O'
            
            # Set minimax player
            minimax_ai.set_player(opponent_player)
            
            while True:
                # Store state information when it's MCTS player's turn
                if current_player == mcts_player:
                    game_history.append((copy.deepcopy(board), current_player))
                
                if current_player == mcts_player:
                    # MCTS player's turn
                    move = self.best_move(board, time_limit=0.5)
                    
                    # Store move information for policy training
                    if game_history:
                        move_index = move[0] * 3 + move[1]
                        policy = torch.zeros(9, device=self.device)
                        policy[move_index] = 1.0
                        
                        # Update the last stored state with move information
                        game_history[-1] = (game_history[-1][0], game_history[-1][1], move_index, policy)
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
                        game_value = 1.0
                    elif winner == "draw":
                        draws += 1
                        game_value = 0.5
                    elif winner:
                        losses += 1
                        game_value = 0.0
                    else:
                        draws += 1  # Full board, no winner
                        game_value = 0.5
                    break
                
                # Switch player
                current_player = 'O' if current_player == 'X' else 'X'
            
            # Process game history for neural network training
            for state, player, move_idx, policy in game_history:
                # Only store MCTS player's moves for training
                if player == mcts_player:
                    # Store training data
                    training_states.append(state)
                    training_policies.append(policy)
                    training_values.append(game_value)
            
            # Train neural network periodically
            if len(training_states) >= 256 or (episode + 1) % 100 == 0:
                if training_states:
                    # Use the latest 2048 examples to avoid memory issues
                    batch_size = min(2048, len(training_states))
                    loss = self.train_policy_network(
                        training_states[-batch_size:],
                        training_policies[-batch_size:],
                        training_values[-batch_size:]
                    )
                    # print(f"Episode {episode+1}, NN Training loss: {loss:.4f}")
                    
                    # Clear training data after using it
                    training_states = []
                    training_policies = []
                    training_values = []
            
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
                
                # Save intermediate model
                if episode % (stats_window * 5) == 0:
                    self.save_model()
                
                # Reset counters
                wins, losses, draws = 0, 0, 0
    
    def train_curriculum(self, episodes=50000):
        """Train using curriculum learning - increasing difficulty."""
        # First step: Pre-train on expert moves
        print("Starting pre-training on expert moves...")
        self.pretrain_on_expert_moves(num_examples=5000, epochs=5)

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
    
    def train_policy_network(self, states, policies, values):
        """Train the policy network from collected experience."""
        # Convert training data to tensors
        state_tensor = torch.stack([self.board_to_tensor(s) for s in states])
        policy_tensor = torch.stack(policies)
        value_tensor = torch.from_numpy(np.array(values)).float().to(self.device)
        
        # Use mixed precision for faster training if available
        scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Forward pass with optional mixed precision
        self.optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                pred_policies, pred_values = self.policy_net(state_tensor)
                policy_loss = nn.functional.cross_entropy(pred_policies, policy_tensor)
                value_loss = nn.functional.mse_loss(pred_values.squeeze(), value_tensor)
                total_loss = policy_loss + value_loss
            
            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            # Original implementation for CPUs or older GPUs
            pred_policies, pred_values = self.policy_net(state_tensor)
            policy_loss = nn.functional.cross_entropy(pred_policies, policy_tensor)
            value_loss = nn.functional.mse_loss(pred_values.squeeze(), value_tensor)
            total_loss = policy_loss + value_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()
        
        return total_loss.item()
    
    def save_model(self, filename=None):
        """Save MCTS statistics, training data and neural network."""
        model_path = get_model_path("mcts") if filename is None else filename
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        data = {
            'position_values': self.position_values,
            'training_stats': self.training_stats,
            'params': {
                'exploration_weight': self.exploration_weight,
                'time_limit': self.time_limit,
                'temperature': self.temperature
            },
            'policy_net': self.policy_net.state_dict()
        }
        
        torch.save(data, model_path)
        print(f"MCTS model saved to {model_path}")
    
    def load_model(self, filename=None):
        """Load MCTS statistics, training data and neural network."""
        try:
            model_path = get_model_path("mcts") if filename is None else filename
            if os.path.exists(model_path):
                data = torch.load(model_path, map_location=self.device)
                
                self.position_values = data.get('position_values', {})
                self.training_stats = data.get('training_stats', {'wins': [], 'losses': [], 'draws': []})
                
                # Optionally load parameters
                params = data.get('params', {})
                self.exploration_weight = params.get('exploration_weight', self.exploration_weight)
                self.time_limit = params.get('time_limit', self.time_limit)
                self.temperature = params.get('temperature', self.temperature)
                
                # Load neural network weights if available
                if 'policy_net' in data:
                    self.policy_net.load_state_dict(data['policy_net'])
                    self.policy_net.to(self.device)  # Ensure it's on the right device
                
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
    
    def guided_simulation(self, board, player, policy):
        """Run a single simulation guided by the provided policy."""
        current_player = player
        board_copy = copy.deepcopy(board)
        move_history = []  # Track moves for RAVE
        
        while self.check_winner(board_copy) is None and self.get_valid_moves(board_copy):
            valid_moves = self.get_valid_moves(board_copy)
            
            # Create a mask for valid moves
            mask = np.zeros(9)
            for r, c in valid_moves:
                mask[r*3 + c] = 1
                
            # Apply mask and renormalize policy
            masked_policy = policy * mask
            policy_sum = masked_policy.sum()
            
            if policy_sum > 0:
                masked_policy = masked_policy / policy_sum
                # Check for immediate winning move
                winning_move = self.find_winning_move(board_copy, current_player)
                if winning_move:
                    move = winning_move
                # Check for blocking move
                else:
                    opponent = 'O' if current_player == 'X' else 'X'
                    blocking_move = self.find_winning_move(board_copy, opponent)
                    if blocking_move:
                        move = blocking_move
                    else:
                        # Sample move proportionally to policy probabilities
                        flat_index = np.random.choice(9, p=masked_policy)
                        move = (flat_index // 3, flat_index % 3)
            else:
                # Fallback to heuristic
                move = self.select_move_heuristic(board_copy, current_player)
            
            # Record move for RAVE statistics
            move_history.append((current_player, move))
            
            # Make the move
            board_copy = self.make_move(board_copy, move, current_player)
            current_player = 'O' if current_player == 'X' else 'X'
        
        result = self.check_winner(board_copy)
        return result, move_history
    
    def simulate(self, board, player):
        """Run a simulation with neural network guidance."""
        current_player = player
        board_copy = copy.deepcopy(board)
        move_history = []  # Track moves for RAVE
        
        while self.check_winner(board_copy) is None and self.get_valid_moves(board_copy):
            # Convert board to tensor for neural network
            board_tensor = self.board_to_tensor(board_copy).unsqueeze(0)  # Add batch dimension
            
            # Get policy and value prediction
            with torch.no_grad():
                policy_probs, value = self.policy_net(board_tensor)
                policy_probs = policy_probs.squeeze().cpu().numpy()
            
            # Create a mask for valid moves
            valid_moves = self.get_valid_moves(board_copy)
            mask = np.zeros(9)
            for r, c in valid_moves:
                mask[r*3 + c] = 1
                
            # Apply mask and renormalize
            policy_probs = policy_probs * mask
            policy_sum = policy_probs.sum()
            
            if policy_sum > 0:
                policy_probs = policy_probs / policy_sum
                # Still check for immediate winning move
                winning_move = self.find_winning_move(board_copy, current_player)
                if winning_move:
                    move = winning_move
                # Check for blocking move
                else:
                    opponent = 'O' if current_player == 'X' else 'X'
                    blocking_move = self.find_winning_move(board_copy, opponent)
                    if blocking_move:
                        move = blocking_move
                    else:
                        # Sample move proportionally to policy probabilities
                        flat_index = np.random.choice(9, p=policy_probs)
                        move = (flat_index // 3, flat_index % 3)
            else:
                # Fallback to original heuristic strategy
                move = self.select_move_heuristic(board_copy, current_player)
                
            # Record move for RAVE statistics
            move_history.append((current_player, move))
            
            # Make the move
            board_copy = self.make_move(board_copy, move, current_player)
            current_player = 'O' if current_player == 'X' else 'X'
        
        result = self.check_winner(board_copy)
        return result, move_history
    
    def batch_simulate(self, boards, players, batch_size=64):
        """Run multiple simulations in parallel using GPU."""
        # Consider GPU memory when setting batch size
        if torch.cuda.is_available():
            # For RTX 3050 with ~4GB VRAM, adjust based on available memory
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024**3)
            # Dynamic batch sizing based on available memory
            adaptive_batch_size = min(int(free_memory_gb * 1000), batch_size)
            batch_size = max(8, adaptive_batch_size)  # Minimum batch size of 8
        
        results = []
        move_histories = []
        
        for i in range(0, len(boards), batch_size):
            batch_boards = boards[i:i+batch_size]
            batch_players = players[i:i+batch_size]
            
            # Convert boards to tensors
            board_tensors = [self.board_to_tensor(board) for board in batch_boards]
            stacked_boards = torch.stack(board_tensors)
            
            # Get policy predictions for all boards at once
            with torch.no_grad():
                policies, values = self.policy_net(stacked_boards)
                policies = policies.cpu().numpy()
                values = values.cpu().numpy()
            
            # Process each board with its policy
            for j, (board, player, policy) in enumerate(zip(batch_boards, batch_players, policies)):
                result, history = self.guided_simulation(board, player, policy)
                results.append(result)
                move_histories.append(history)
        
        return results, move_histories
    
    def select_move_heuristic(self, board, player):
        """Fallback heuristic move selection."""
        moves = self.get_valid_moves(board)
        
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
        for i, (move, weight) in enumerate(zip(moves, weights)):
            current_weight += weight
            if current_weight >= selection:
                return move
        
        return random.choice(moves)  # Fallback
    
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
    
    def board_to_tensor(self, board):
        """Convert the board to a tensor representation."""
        # 3 channels: player pieces, opponent pieces, and empty spaces
        tensor_board = torch.zeros(3, 3, 3, device=self.device)
        
        opponent = 'X' if self.player == 'O' else 'O'
        
        for i in range(3):
            for j in range(3):
                if board[i][j] == self.player:
                    tensor_board[0, i, j] = 1.0
                elif board[i][j] == opponent:
                    tensor_board[1, i, j] = 1.0
                elif board[i][j] is None:
                    tensor_board[2, i, j] = 1.0
                    
        return tensor_board
    

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