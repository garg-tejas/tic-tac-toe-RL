class MinimaxAI:
    def __init__(self):
        self.player = 'O'  # Default player
        self.opponent = 'X'
    
    def evaluate(self, board):
        # Check rows
        for row in range(3):
            if board[row][0] == board[row][1] == board[row][2]:
                if board[row][0] == self.player: return 10
                if board[row][0] == self.opponent: return -10
        
        # Check columns
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col]:
                if board[0][col] == self.player: return 10
                if board[0][col] == self.opponent: return -10
        
        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2]:
            if board[0][0] == self.player: return 10
            if board[0][0] == self.opponent: return -10
        
        if board[0][2] == board[1][1] == board[2][0]:
            if board[0][2] == self.player: return 10
            if board[0][2] == self.opponent: return -10
        
        return 0  # No winner yet
    
    def is_board_full(self, board):
        return all(cell is not None for row in board for cell in row)
    
    def evaluate_heuristic(self, board):
        """Evaluate board state with a simple heuristic when depth limit is reached."""
        # Count favorable positions for each player
        player_score = self.count_favorable_positions(board, self.player)
        opponent_score = self.count_favorable_positions(board, self.opponent)
        
        # Return difference as normalized score
        return (player_score - opponent_score) / 10.0
    
    def count_favorable_positions(self, board, player):
        """Count positions that favor a player."""
        score = 0
        
        # Prefer center
        if board[1][1] == player:
            score += 3
        
        # Prefer corners
        corners = [(0,0), (0,2), (2,0), (2,2)]
        for r, c in corners:
            if board[r][c] == player:
                score += 2
        
        # Check for two-in-a-row with empty third position
        # Rows
        for row in range(3):
            row_values = [board[row][col] for col in range(3)]
            if row_values.count(player) == 2 and row_values.count(None) == 1:
                score += 5
        
        # Columns
        for col in range(3):
            col_values = [board[row][col] for row in range(3)]
            if col_values.count(player) == 2 and col_values.count(None) == 1:
                score += 5
        
        # Diagonals
        diag1 = [board[0][0], board[1][1], board[2][2]]
        if diag1.count(player) == 2 and diag1.count(None) == 1:
            score += 5
            
        diag2 = [board[0][2], board[1][1], board[2][0]]
        if diag2.count(player) == 2 and diag2.count(None) == 1:
            score += 5
        
        return score
    
    def minimax(self, board, depth, is_maximizing, depth_limit=float('inf'), alpha=-float('inf'), beta=float('inf')):
        score = self.evaluate(board)
        
        # Terminal states
        if score == 10:
            return score - depth  # Win (prefer quicker wins)
        if score == -10:
            return score + depth  # Loss (prefer longer losses)
        if self.is_board_full(board):
            return 0
        
        # Depth limit reached - use heuristic evaluation instead
        if depth >= depth_limit:
            return self.evaluate_heuristic(board)
        
        if is_maximizing:
            best = -float('inf')
            for row in range(3):
                for col in range(3):
                    if board[row][col] is None:
                        board[row][col] = self.player
                        best = max(best, self.minimax(board, depth + 1, False, depth_limit, alpha, beta))
                        board[row][col] = None
                        
                        # Alpha-Beta pruning
                        alpha = max(alpha, best)
                        if beta <= alpha:
                            break
            return best
        else:
            best = float('inf')
            for row in range(3):
                for col in range(3):
                    if board[row][col] is None:
                        board[row][col] = self.opponent
                        best = min(best, self.minimax(board, depth + 1, True, depth_limit, alpha, beta))
                        board[row][col] = None
                        
                        # Alpha-Beta pruning
                        beta = min(beta, best)
                        if beta <= alpha:
                            break
            return best
    
    def best_move(self, board, depth_limit=float('inf')):
        best_val = -float('inf')
        move = None
        
        # Default to 'O' if player isn't set
        if not hasattr(self, 'player'):
            self.player = 'O'
            self.opponent = 'X'
        
        # Quick check for winning moves first (optimization)
        for row in range(3):
            for col in range(3):
                if board[row][col] is None:
                    board[row][col] = self.player
                    if self.evaluate(board) == 10:
                        board[row][col] = None
                        return (row, col)  # Return immediate winning move
                    board[row][col] = None
        
        # Quick check for blocking moves
        for row in range(3):
            for col in range(3):
                if board[row][col] is None:
                    board[row][col] = self.opponent
                    if self.evaluate(board) == -10:
                        board[row][col] = None
                        return (row, col)  # Return immediate blocking move
                    board[row][col] = None
        
        # Apply depth-limited minimax if no immediate win/block
        for row in range(3):
            for col in range(3):
                if board[row][col] is None:
                    board[row][col] = self.player
                    move_val = self.minimax(board, 0, False, depth_limit)
                    board[row][col] = None
                    
                    if move_val > best_val:
                        best_val = move_val
                        move = (row, col)
                        
        return move
    
    def set_player(self, player):
        """Set the player marker (X or O)."""
        self.player = player
        self.opponent = 'X' if player == 'O' else 'O'