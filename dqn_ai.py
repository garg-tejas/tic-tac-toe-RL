import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os
from utils import get_model_path, get_plot_path

class DQN(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, output_size=9):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
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
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.training_stats = {'wins': [], 'losses': [], 'draws': []}  # For monitoring progress

        self.training = True  # Flag for training mode vs evaluation mode
        self.player = "O"
        self.opponent = 'X'
        self.load_model()  # Load existing model if available

    def set_training_mode(self, is_training=True):
        """Toggle between training and evaluation modes."""
        self.training = is_training
        if is_training:
            self.policy_net.train()
        else:
            self.policy_net.eval()

    def set_player(self, player):
        """Set the player marker (X or O)."""
        self.player = player
        self.opponent = 'X' if player == 'O' else 'O'
        
    def get_state(self, board):
        """Enhanced state representation with robust board parsing."""
        # Handle 2D board
        if isinstance(board, list) and all(isinstance(row, list) for row in board):
            # Convert 2D board to 1D for canonicalization
            board_1d = [" " if cell is None else cell for row in board for cell in row]
        # Handle string representation
        elif isinstance(board, str):
            try:
                # Try to parse string as tuple representation
                import ast
                board_1d = list(ast.literal_eval(board))
            except:
                board_1d = list(board)  # Direct conversion
        # Handle tuple
        elif isinstance(board, tuple):
            board_1d = list(board)
        # Default: assume it's already 1D
        else:
            board_1d = board
            
        # Get canonical state
        canonical_board = self.get_canonical_state(board_1d)
        
        # Basic board state
        basic_state = [
            1.0 if cell == self.player else 
            -1.0 if cell != " " else 
            0.0 for cell in canonical_board
        ]
        
        # Add feature: player's two-in-a-rows
        player_two_in_rows = self.count_two_in_a_row(canonical_board, self.player)
        
        # Add feature: opponent's two-in-a-rows
        opponent = 'X' if self.player == 'O' else 'O'
        opponent_two_in_rows = self.count_two_in_a_row(canonical_board, opponent)
        
        # Add feature: center control
        center_control = 1.0 if canonical_board[4] == self.player else -1.0 if canonical_board[4] == opponent else 0.0
        
        # Combine all features
        enhanced_state = basic_state + [player_two_in_rows/3.0, opponent_two_in_rows/3.0, center_control]
        
        return torch.tensor(enhanced_state, dtype=torch.float32, device=self.device).unsqueeze(0)
    
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

    def soft_update_target_network(self, tau=0.01):
        """Soft update target network: θ_target = τ*θ_policy + (1-τ)*θ_target"""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                            self.policy_net.parameters()):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Ensure consistent tensor dimensions 
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.bool)
        
        # Current Q Values - Add dimension safety with squeeze(1)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Improved Double DQN with stability checks
        with torch.no_grad():
            # Get actions from policy net for next states
            policy_next_qvals = self.policy_net(next_states)
            
            # Add noise for regularization (optional)
            if self.training:
                noise = torch.randn_like(policy_next_qvals) * 0.1
                policy_next_qvals += noise
                
            # Get best actions
            next_actions = policy_next_qvals.max(1)[1].unsqueeze(1)
            
            # Get Q-values from target net for those actions - Add dimension safety with squeeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            
            # Apply reward clipping for stability
            clamped_rewards = torch.clamp(rewards, -10, 10)
            
            # Compute target Q values
            target_q_values = clamped_rewards + (self.gamma * next_q_values * ~dones)
            
            # Additional stability: ensure targets are within reasonable range
            target_q_values = torch.clamp(target_q_values, -15, 15)
        
        # Compute loss with Huber loss (more stable than MSE)
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)  # Lower value
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def pretrain_on_expert_moves(self, num_examples=10000, epochs=10):
        """Pre-train the DQN network on expert Tic-Tac-Toe moves."""
        print("Pre-training DQN on expert moves...")
        
        # Containers for training data
        training_states = []
        training_targets = []
        
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
                if self.check_winner(board):
                    break
            
            # If game is already over, skip this example
            winner = self.check_winner(board)
            if winner:
                continue
            
            # Now find the "expert move" for this position
            best_move = self.get_expert_move(board, current_player)
            if best_move is not None:
                # Create target Q-values - high for best move, low for others
                state = self.get_state(board)
                
                # Get current Q-value predictions as starting point
                with torch.no_grad():
                    q_values = self.policy_net(state).squeeze().cpu().numpy()
                
                # Modify Q-values to favor expert move
                for action in range(9):
                    if action == best_move:
                        q_values[action] = 10.0  # High value for best action
                    elif board[action] != " ":
                        q_values[action] = -10.0  # Invalid move
                    else:
                        q_values[action] = 0.0   # Neutral for other valid moves
                
                # Store the example
                training_states.append(state)
                training_targets.append(torch.tensor(q_values, device=self.device))
        
        # Train for multiple epochs on this data
        print(f"Generated {len(training_states)} expert examples")
        for epoch in range(epochs):
            # Shuffle the data
            indices = list(range(len(training_states)))
            random.shuffle(indices)
            shuffled_states = [training_states[i] for i in indices]
            shuffled_targets = [training_targets[i] for i in indices]
            
            # Train in batches
            batch_size = 64
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(shuffled_states), batch_size):
                batch_states = torch.cat(shuffled_states[i:i+batch_size])
                batch_targets = torch.stack(shuffled_targets[i:i+batch_size])
                
                # Forward pass
                self.optimizer.zero_grad()
                predicted = self.policy_net(batch_states)
                
                # Calculate loss
                loss = self.loss_fn(predicted, batch_targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(1, num_batches)
            print(f"Pre-training epoch {epoch+1}/{epochs}, Avg loss: {avg_loss:.4f}")
        
        # Update target network with pre-trained weights
        self.update_target_network()
        print("DQN pre-training complete")

    def get_expert_move(self, board, player):
        """Determine the best move according to expert Tic-Tac-Toe strategy."""
        # Convert format if needed (handling both list and string representations)
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

    def train(self, episodes=50000, target_update=1000):
        """Train the agent through self-play."""
        wins, losses, draws = 0, 0, 0
        stats_window = 1000
    
        for episode in range(episodes):
            board = [" "] * 9
            current_player = "X"  # Start with X
            done = False
            
            # For tracking experiences
            game_history = []
            
            while not done:
                state = self.get_state(board)
                
                # Choose action for current player
                if current_player == self.player:
                    action = self.choose_action(board)
                else:
                    # When self-playing, use same network but with exploration for opponent
                    temp_epsilon = self.epsilon  # Store current epsilon
                    self.epsilon = min(0.3, self.epsilon * 2)  # More exploration for opponent
                    action = self.choose_action(board)
                    self.epsilon = temp_epsilon  # Restore epsilon
                
                # Make move
                board[action] = current_player
                next_state = self.get_state(board)
                
                # Get reward from current player's perspective
                if current_player == self.player:
                    reward, done = self.reward_function(board)
                else:
                    # Opponent's perspective
                    opponent_reward, done = self.reward_function(board)
                    reward = -opponent_reward  # Invert reward for our agent
                
                # Store experience tuple (will learn from these later)
                game_history.append((state, action, reward, next_state, done, current_player))
                
                # Update game statistics if done
                if done:
                    winner = self.check_winner(board)
                    if winner == self.player:
                        wins += 1
                    elif winner == "draw":
                        draws += 1
                    else:
                        losses += 1
                    break
                    
                # Switch player
                current_player = "O" if current_player == "X" else "X"
            
            # Learn from experiences after game ends
            for exp_state, exp_action, exp_reward, exp_next_state, exp_done, exp_player in game_history:
                # Only learn from agent's moves or adjust rewards for opponent moves
                if exp_player == self.player:
                    self.store_experience(exp_state, exp_action, exp_reward, exp_next_state, exp_done)
                else:
                    # For opponent moves, we can either:
                    # 1. Skip them (don't learn)
                    # 2. Invert rewards and learn (creates a more robust agent)
                    self.store_experience(exp_state, exp_action, -exp_reward, exp_next_state, exp_done)
            
            # Training step
            self.train_step()
            
            # Update target network periodically
            if episode % target_update == 0:
                self.update_target_network()
            
            # Apply soft update every few episodes instead of every episode
            if episode % 5 == 0:
                self.soft_update_target_network(tau=0.01)

            # Step the scheduler periodically
            if episode % 100 == 0:
                self.scheduler.step()
            
            # Decay epsilon (just once)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Stats reporting
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

    def reward_function(self, board, perspective_player=None):
        """Enhanced reward function with player perspective."""
        # Use provided perspective or default to agent's player
        perspective = perspective_player if perspective_player else self.player
        opponent = 'X' if perspective == 'O' else 'O'
        
        winner = self.check_winner(board)
        
        if winner == perspective:
            return 10, True  # Win
        elif winner == "draw":
            return 1, True   # Draw
        elif winner is not None:
            return -10, True  # Loss
        
        # Strategic rewards from the perspective player's viewpoint
        score = 0
        
        # Reward for creating two-in-a-row opportunities
        score += self.count_two_in_a_row(board, perspective) * 0.5
        
        # Penalty for allowing opponent two-in-a-row
        score -= self.count_two_in_a_row(board, opponent) * 0.7
        
        # Reward for taking center
        if board[4] == perspective:  # Center position
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
        """Enhanced curriculum learning with model persistence between stages."""
        # Ensure consistent player assignment
        self.set_player("O")  # Agent plays as O
        
        # Stage 1: Expert pre-training
        print("Stage 1: Pre-training on expert moves...")
        self.pretrain_on_expert_moves(num_examples=5000, epochs=5)
        self.save_model("dqn_stage1.pth")  # Save after each stage
        
        # Stage 2: Training against random
        print("Stage 2: Training against random opponent...")
        self.train_against_random(int(episodes * 0.35))
        self.save_model("dqn_stage2.pth")
        
        # Stage 3: Training against easy minimax
        print("Stage 3: Training against easy minimax...")
        self.train_against_minimax(int(episodes * 0.15), depth_limit=1)
        self.save_model("dqn_stage3.pth")
        
        # Stage 4: Training against harder minimax
        print("Stage 4: Training against harder minimax...")
        self.train_against_minimax(int(episodes * 0.25), depth_limit=3)
        self.save_model("dqn_stage4.pth")
        
        # Stage 5: Self-play
        print("Stage 5: Training with full self-play...")
        self.train(int(episodes * 0.25))
        self.save_model()  # Final save
        
        print("Curriculum training complete!")

    def train_against_random(self, episodes=5000):
        """Train against a random opponent."""
        opponent_player = 'X' if self.player == 'O' else 'O'
        wins, losses, draws = 0, 0, 0
        stats_window = 1000  # Number of episodes between stats updates
    
        for episode in range(episodes):
            board = [" "] * 9
            done = False
            
            while not done:
                # Store the initial state before agent's move
                current_state = self.get_state(board)
    
                # DQN agent's turn
                action = self.choose_action(board)
                board[action] = self.player
    
                # Get state after agent's move
                agent_next_state = self.get_state(board)
                reward, done = self.reward_function(board)
    
                # Store experience just for the agent's move
                self.store_experience(current_state, action, reward, agent_next_state, done)
                
                # Check if game ended after agent's move
                if done:
                    winner = self.check_winner(board)
                    if winner == self.player:
                        wins += 1
                    elif winner == "draw":
                        draws += 1
                    else:
                        losses += 1
                else:
                    # Random opponent's turn
                    valid_moves = self.get_valid_moves(board)
                    opponent_action = random.choice(valid_moves)
                    board[opponent_action] = opponent_player
                    
                    # Check if game ended after opponent's move
                    reward, done = self.reward_function(board)
                    if done:
                        winner = self.check_winner(board)
                        if winner == self.player:
                            wins += 1
                        elif winner == "draw":
                            draws += 1
                        else:
                            losses += 1
                
                # Training step after completing the current game state
                self.train_step()
            
            # Apply soft update every few episodes instead of every episode
            if episode % 5 == 0:
                self.soft_update_target_network(tau=0.01)
    
            # Decay epsilon every 100 episodes
            if episode % 100 == 0:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
            # Calculate and report statistics every 1000 episodes
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
            done = False
            
            while not done:
                # Store current state before agent's move
                current_state = self.get_state(board)
                
                # DQN agent's turn
                action = self.choose_action(board)
                board[action] = self.player
                
                # Get state after agent's move
                agent_next_state = self.get_state(board)
                reward, done = self.reward_function(board)
                
                # Store experience just for the agent's move
                self.store_experience(current_state, action, reward, agent_next_state, done)
                
                # Check if game ended after agent's move
                if done:
                    winner = self.check_winner(board)
                    if winner == self.player:
                        wins += 1
                    elif winner == "draw":
                        draws += 1
                    else:
                        losses += 1
                else:
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
                        
                        # Check if game ended after opponent's move
                        reward, done = self.reward_function(board)
                        if done:
                            winner = self.check_winner(board)
                            if winner == self.player:
                                wins += 1
                            elif winner == "draw":
                                draws += 1
                            else:
                                losses += 1
                
                # Training step after completing the current game state
                self.train_step()
            
            # Apply soft update every few episodes instead of every episode
            if episode % 5 == 0:
                self.soft_update_target_network(tau=0.01)
    
            # Decay epsilon every 100 episodes
            if episode % 100 == 0:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Calculate and report statistics every 1000 episodes
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
            plt.title('DQN Agent Learning Progress', fontsize=16, pad=20)
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
            plt.savefig(get_plot_path("dqn"), dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError as e:
            print(f"Required plotting libraries not available: {e}")
            print("Install matplotlib, pandas, and seaborn for visualization.")