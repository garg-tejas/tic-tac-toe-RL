class TicTacToe:
    def __init__(self):
        # Initialize an empty 3x3 board
        self.board = [[None for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'  # X goes first
        self.winner = None
        self.history = []
        self.redo_stack = []
        
    def reset_game(self):
        # Reset the game state
        self.board = [[None for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.winner = None
        
    def make_move(self, row, col):
        # Save current state before making move
        if 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] is None:
            # Save state for undo
            current_state = {
                'board': [row[:] for row in self.board],  # Deep copy
                'player': self.current_player,
                'winner': self.winner
            }
            self.history.append(current_state)
            self.redo_stack = []  # Clear redo stack on new move
            
            # Make move
            self.board[row][col] = self.current_player
            self.check_winner()
            
            # Switch players if game isn't over
            if not self.winner:
                self.current_player = 'O' if self.current_player == 'X' else 'X'
                
            return True
        return False
    
    def undo(self):
        if len(self.history) > 0:
            # Save current state for redo
            current_state = {
                'board': [row[:] for row in self.board],
                'player': self.current_player,
                'winner': self.winner
            }
            self.redo_stack.append(current_state)
            
            # Restore previous state
            prev_state = self.history.pop()
            self.board = prev_state['board']
            self.current_player = prev_state['player']
            self.winner = prev_state['winner']
            return True
        return False
    
    def redo(self):
        if len(self.redo_stack) > 0:
            # Save current state for undo
            current_state = {
                'board': [row[:] for row in self.board],
                'player': self.current_player,
                'winner': self.winner
            }
            self.history.append(current_state)
            
            # Restore redo state
            next_state = self.redo_stack.pop()
            self.board = next_state['board']
            self.current_player = next_state['player']
            self.winner = next_state['winner']
            return True
        return False
        
    def check_winner(self):
        # Check rows
        for row in range(3):
            if self.board[row][0] == self.board[row][1] == self.board[row][2] and self.board[row][0]:
                self.winner = self.board[row][0]
                return
                
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] and self.board[0][col]:
                self.winner = self.board[0][col]
                return
                
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0]:
            self.winner = self.board[0][0]
            return
            
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2]:
            self.winner = self.board[0][2]
            return
            
        # Check for draw
        if all(cell for row in self.board for cell in row):
            self.winner = "Draw"
            
    def get_available_moves(self):
        # Return list of available (row, col) moves
        moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] is None:
                    moves.append((row, col))
        return moves
        
    def is_game_over(self):
        return self.winner is not None