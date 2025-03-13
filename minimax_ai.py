class MinimaxAI:
    def __init__(self):
        pass
    
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
    
    def minimax(self, board, depth, is_maximizing):
        score = self.evaluate(board)
        
        # Terminal states
        if score == 10 or score == -10:
            return score - depth
        if self.is_board_full(board):
            return 0
        
        if is_maximizing:
            best = -float('inf')
            for row in range(3):
                for col in range(3):
                    if board[row][col] is None:
                        board[row][col] = 'O'
                        best = max(best, self.minimax(board, depth + 1, False))
                        board[row][col] = None
            return best
        else:
            best = float('inf')
            for row in range(3):
                for col in range(3):
                    if board[row][col] is None:
                        board[row][col] = 'X'
                        best = min(best, self.minimax(board, depth + 1, True))
                        board[row][col] = None
            return best
    
    def best_move(self, board):
        best_val = -float('inf')
        move = None
        
        # Default to 'O' if player isn't set
        if not hasattr(self, 'player'):
            self.player = 'O'
            self.opponent = 'X'
        
        for row in range(3):
            for col in range(3):
                if board[row][col] is None:
                    board[row][col] = self.player
                    move_val = self.minimax(board, 0, False)
                    board[row][col] = None
                    
                    if move_val > best_val:
                        best_val = move_val
                        move = (row, col)
                        
        return move
    
    def set_player(self, player):
        """Set the player marker (X or O)."""
        self.player = player
        self.opponent = 'X' if player == 'O' else 'O'
        print(f"Minimax AI will play as {player}")