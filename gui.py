import pygame
import json
import os
from game import TicTacToe
from minimax_ai import MinimaxAI
from q_learning_ai import QLearningTicTacToe
from mcts_ai import MCTSAI
from dqn_ai import DQNTicTacToe
import random

class TicTacToeGUI:
    # Constants
    WIDTH, HEIGHT = 800, 800
    LINE_WIDTH = 15
    BOARD_ROWS, BOARD_COLS = 3, 3
    CELL_SIZE = 200
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (220, 50, 50)
    BLUE = (50, 120, 220)
    LIGHT_BLUE = (100, 170, 255)
    LIGHT_RED = (255, 130, 130)
    GRAY = (200, 200, 200)
    DARK_GRAY = (80, 80, 80)
    BG_COLOR = (240, 240, 245)
    HIGHLIGHT = (170, 255, 170)
    
    # AI Options
    AI_OPTIONS = ["Minimax", "Q-Learning", "DQN", "MCTS"]
    
    def __init__(self):
        pygame.init()
        
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe AI")
        self.font = pygame.font.Font(None, 36)
        self.title_font = pygame.font.Font(None, 60)
        self.small_font = pygame.font.Font(None, 24)
        
        self.game = TicTacToe()
        self.selected_ai = None
        self.minimax_ai = MinimaxAI()
        self.q_learning_ai = QLearningTicTacToe()  # Initialize Q-Learning AI
    
        self.mcts_ai = MCTSAI(exploration_weight=1.41, time_limit=1, temperature=0.1)
    
        self.dqn_ai = DQNTicTacToe()
        self.dqn_ai.load_model()  # Load pre-trained model

        # Animation tracking
        self.animations = []
        self.win_animation = None
        
        # Initialize scoreboard
        self.scores = {'X': 0, 'O': 0, 'Draw': 0}
        self.score_updated = False
        self.load_scores()
    
        self.player_marker = 'X'  # Default
        self.game_mode = "AI"     # Default
        
    def load_scores(self):
        try:
            if os.path.exists('scores.json'):
                with open('scores.json', 'r') as f:
                    self.scores = json.load(f)
        except:
            # If file doesn't exist or is corrupted, use default scores
            self.scores = {'X': 0, 'O': 0, 'Draw': 0}
        
    def save_scores(self):
        with open('scores.json', 'w') as f:
            json.dump(self.scores, f)
        
    def update_scoreboard(self):
        if self.game.winner:
            if self.game.winner == 'Draw':
                self.scores['Draw'] += 1
            else:
                self.scores[self.game.winner] += 1
            self.save_scores()
            self.score_updated = True
        
    def draw_scoreboard(self):
        # Draw scoreboard container
        board_width = 200
        board_height = 150
        x = self.WIDTH // 2 - board_width // 2
        y = 20
        
        pygame.draw.rect(self.screen, self.WHITE, (x, y, board_width, board_height), 0, border_radius=10)
        pygame.draw.rect(self.screen, self.DARK_GRAY, (x, y, board_width, board_height), 2, border_radius=10)
        
        # Draw title
        title_text = self.font.render("SCOREBOARD", True, self.BLACK)
        self.screen.blit(title_text, (x + board_width//2 - title_text.get_width()//2, y + 15))
        
        # Draw scores with dynamic labels
        y_offset = 50
        if self.game_mode == "Human":
            player_x_text = "Player 1 (X)"
            player_o_text = "Player 2 (O)"
        else:  # AI mode
            if self.player_marker == 'X':
                player_x_text = "You (X)"
                player_o_text = "AI (O)"
            else:
                player_x_text = "AI (X)"
                player_o_text = "You (O)"
        
        score_text = self.font.render(f"{player_x_text}: {self.scores['X']}", True, self.RED)
        self.screen.blit(score_text, (x + 20, y + y_offset))
        
        y_offset += 30
        score_text = self.font.render(f"{player_o_text}: {self.scores['O']}", True, self.BLUE)
        self.screen.blit(score_text, (x + 20, y + y_offset))
        
        y_offset += 30
        score_text = self.font.render(f"Draws: {self.scores['Draw']}", True, self.GRAY)
        self.screen.blit(score_text, (x + 20, y + y_offset))
        
    def draw_button(self, text, x, y, width, height, default_color, hover_color):
        mouse_pos = pygame.mouse.get_pos()
        button_rect = pygame.Rect(x, y, width, height)
        
        if button_rect.collidepoint(mouse_pos):
            color = hover_color
            pygame.draw.rect(self.screen, color, button_rect, 0, border_radius=10)
            pygame.draw.rect(self.screen, self.DARK_GRAY, button_rect, 2, border_radius=10)
        else:
            color = default_color
            pygame.draw.rect(self.screen, color, button_rect, 0, border_radius=10)
            pygame.draw.rect(self.screen, self.BLACK, button_rect, 2, border_radius=10)
            
        text_surf = self.font.render(text, True, self.BLACK)
        text_rect = text_surf.get_rect(center=button_rect.center)
        self.screen.blit(text_surf, text_rect)
        
        return button_rect

    def draw_menu(self):
        self.screen.fill(self.BG_COLOR)
        
        pygame.draw.rect(self.screen, self.LIGHT_BLUE, (100, 50, self.WIDTH-200, 100), 0, border_radius=15)
        
        shadow_text = self.title_font.render("TIC-TAC-TOE", True, self.DARK_GRAY)
        self.screen.blit(shadow_text, (self.WIDTH//4+2, 62))
        title_text = self.title_font.render("TIC-TAC-TOE", True, self.WHITE)
        self.screen.blit(title_text, (self.WIDTH//4, 60))
        
        subtitle = self.font.render("Select AI Opponent", True, self.BLACK)
        self.screen.blit(subtitle, (self.WIDTH//3, 160))
        
        button_rects = []
        for i, ai in enumerate(self.AI_OPTIONS):
            button_rect = self.draw_button(ai, 250, 220 + i * 100, 300, 60, 
                                     self.LIGHT_BLUE if i % 2 == 0 else self.LIGHT_RED, 
                                     self.HIGHLIGHT)
            button_rects.append(button_rect)
        
        pygame.display.update()
        return button_rects

    def main_menu(self):
        # First choose game mode
        if not self.choose_game_mode():
            return False
            
        # If AI mode, choose AI opponent
        if self.game_mode == "AI":
            if not self.choose_ai():
                return False
        
        # Let player choose marker (X or O)
        if not self.choose_marker():
            return False
        
        # If player chose 'O', let AI make the first move
        if self.game_mode == "AI" and self.player_marker == 'O':
            # Delay slightly for UX
            pygame.time.delay(800)  # Increased delay
            
            try:
                if self.selected_ai == "Minimax":
                    ai_row, ai_col = self.minimax_ai.best_move(self.game.board, depth_limit=4)
                elif self.selected_ai == "Q-Learning":
                    ai_row, ai_col = self.q_learning_ai.best_move(self.game.board)
                elif self.selected_ai == "DQN" and self.dqn_ai:
                    ai_row, ai_col = self.dqn_ai.best_move(self.game.board)
                elif self.selected_ai == "MCTS" and self.mcts_ai:
                    # For MCTS, set the player and pass time limit
                    self.mcts_ai.set_player('X' if self.player_marker == 'O' else 'O')
                    ai_row, ai_col = self.mcts_ai.best_move(self.game.board, time_limit=1.0)
                else:  # Default to random if AI is not properly initialized
                    valid_moves = self.game.get_available_moves()
                    ai_row, ai_col = random.choice(valid_moves) if valid_moves else (0, 0)
            except Exception as e:
                print(f"AI error: {e}. Using random fallback.")
                valid_moves = self.game.get_available_moves()
                ai_row, ai_col = random.choice(valid_moves) if valid_moves else (0, 0)
                
            # Make AI's move
            if self.game.make_move(ai_row, ai_col):
                # Create animation for AI move
                cell_width = (self.WIDTH-200) // self.BOARD_COLS
                cell_height = (self.HEIGHT-350) // self.BOARD_ROWS
                center_x = ai_col * cell_width + 100 + cell_width // 2
                center_y = ai_row * cell_height + 200 + cell_height // 2
                    
                size = min(cell_width, cell_height) // 2 - 15
                self.animations.append(
                    MarkAnimation(
                        self.game.board[ai_row][ai_col],
                        center_x,
                        center_y,
                        size
                    )
                )
        
        return True
    
    def choose_marker(self):
        running = True
        while running:
            self.screen.fill(self.BG_COLOR)
            
            # Title
            title = self.title_font.render("Choose Your Marker", True, self.BLACK)
            self.screen.blit(title, (self.WIDTH//2 - title.get_width()//2, 150))
            
            # Buttons
            x_rect = self.draw_button("Play as X", self.WIDTH//2 - 250, 300, 200, 100, self.RED, self.HIGHLIGHT)
            o_rect = self.draw_button("Play as O", self.WIDTH//2 + 50, 300, 200, 100, self.BLUE, self.HIGHLIGHT)
            
            pygame.display.update()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if x_rect.collidepoint(pos):
                        self.player_marker = 'X'
                        # Set AI markers to O
                        if hasattr(self, 'dqn_ai') and self.dqn_ai:
                            self.dqn_ai.player = 'O'
                        self.q_learning_ai.player = 'O' 
                        running = False
                    elif o_rect.collidepoint(pos):
                        self.player_marker = 'O'
                        # Set AI markers to X
                        if hasattr(self, 'dqn_ai') and self.dqn_ai:
                            self.dqn_ai.player = 'X'
                        self.q_learning_ai.player = 'X'
                        running = False
        return True
    
    def choose_game_mode(self):
        running = True
        while running:
            self.screen.fill(self.BG_COLOR)
            
            # Title
            title = self.title_font.render("Game Mode", True, self.BLACK)
            self.screen.blit(title, (self.WIDTH//2 - title.get_width()//2, 150))
            
            # Buttons
            ai_rect = self.draw_button("Play vs AI", self.WIDTH//2 - 250, 300, 200, 100, self.LIGHT_BLUE, self.HIGHLIGHT)
            human_rect = self.draw_button("Play vs Human", self.WIDTH//2 + 50, 300, 200, 100, self.LIGHT_RED, self.HIGHLIGHT)
            
            pygame.display.update()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if ai_rect.collidepoint(pos):
                        self.game_mode = "AI"
                        running = False
                    elif human_rect.collidepoint(pos):
                        self.game_mode = "Human"
                        running = False
        
        return True
    
    def choose_ai(self):
        running = True
        button_rects = self.draw_menu()  # Draws AI selection menu
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    for i, rect in enumerate(button_rects):
                        if rect.collidepoint(mouse_pos):
                            self.selected_ai = self.AI_OPTIONS[i]
                            running = False
                            
            pygame.display.update()
        
        return True

    def draw_board(self):
        self.screen.fill(self.BG_COLOR)
        
        board_rect = pygame.Rect(100, 200, self.WIDTH-200, self.HEIGHT-350)
        pygame.draw.rect(self.screen, self.WHITE, board_rect, 0, border_radius=15)
        pygame.draw.rect(self.screen, self.DARK_GRAY, board_rect, 3, border_radius=15)
        
        for row in range(1, self.BOARD_ROWS):
            y_pos = row * (self.HEIGHT-350) // self.BOARD_ROWS + 200
            pygame.draw.line(self.screen, self.DARK_GRAY, (110, y_pos), (self.WIDTH-110, y_pos), self.LINE_WIDTH//2)
        for col in range(1, self.BOARD_COLS):
            x_pos = col * (self.WIDTH-200) // self.BOARD_COLS + 100
            pygame.draw.line(self.screen, self.DARK_GRAY, (x_pos, 210), (x_pos, self.HEIGHT-140), self.LINE_WIDTH//2)
        
        status_text = f"AI: {self.selected_ai}"
        status_surf = self.font.render(status_text, True, self.BLACK)
        self.screen.blit(status_surf, (100, self.HEIGHT-120))
        
        if self.game.winner is None:
            turn_text = f"Current Turn: {'X' if self.game.current_player == 'X' else 'O'}"
            turn_text += f" ({('You' if self.player_marker == self.game.current_player else 'AI')})" if self.game_mode == "AI" else ""            
            turn_color = self.RED if self.game.current_player == 'X' else self.BLUE
            turn_surf = self.font.render(turn_text, True, turn_color)
            self.screen.blit(turn_surf, (self.WIDTH//2 - turn_surf.get_width()//2, self.HEIGHT-120))
        
        # Add undo/redo buttons
        undo_rect = self.draw_button("Undo", 100, self.HEIGHT-70, 100, 40, self.LIGHT_RED, self.HIGHLIGHT)
        redo_rect = self.draw_button("Redo", self.WIDTH-200, self.HEIGHT-70, 100, 40, self.LIGHT_BLUE, self.HIGHLIGHT)
        
        # New game button remains centered
        new_game_rect = self.draw_button("New Game", self.WIDTH//2-100, self.HEIGHT-70, 200, 40, self.LIGHT_BLUE, self.HIGHLIGHT)
        
        return new_game_rect, undo_rect, redo_rect

    def draw_marks(self):
        cell_width = (self.WIDTH-200) // self.BOARD_COLS
        cell_height = (self.HEIGHT-350) // self.BOARD_ROWS
        
        # Update and remove completed animations
        active_animations = []
        for anim in self.animations:
            anim.update()
            if not anim.complete:
                active_animations.append(anim)
        self.animations = active_animations
        
        # Draw static marks for cells not being animated
        animated_cells = set()
        for anim in self.animations:
            for row in range(self.BOARD_ROWS):
                for col in range(self.BOARD_COLS):
                    center_x = col * cell_width + 100 + cell_width // 2
                    center_y = row * cell_height + 200 + cell_height // 2
                    if abs(center_x - anim.center_x) < 5 and abs(center_y - anim.center_y) < 5:
                        animated_cells.add((row, col))
        
        # Draw static marks
        for row in range(self.BOARD_ROWS):
            for col in range(self.BOARD_COLS):
                if (row, col) in animated_cells:
                    continue
                
                # Calculate center coordinates for this cell
                center_x = col * cell_width + 100 + cell_width // 2
                center_y = row * cell_height + 200 + cell_height // 2
                        
                if self.game.board[row][col] == 'X':
                    offset = 5
                    size = min(cell_width, cell_height) // 2 - 15
                    
                    pygame.draw.line(self.screen, self.DARK_GRAY, 
                                    (center_x - size + offset, center_y - size + offset), 
                                    (center_x + size + offset, center_y + size + offset), 
                                    self.LINE_WIDTH)
                    pygame.draw.line(self.screen, self.DARK_GRAY, 
                                    (center_x + size + offset, center_y - size + offset), 
                                    (center_x - size + offset, center_y + size + offset), 
                                    self.LINE_WIDTH)
                    
                    pygame.draw.line(self.screen, self.RED, 
                                    (center_x - size, center_y - size), 
                                    (center_x + size, center_y + size), 
                                    self.LINE_WIDTH)
                    pygame.draw.line(self.screen, self.RED, 
                                    (center_x + size, center_y - size), 
                                    (center_x - size, center_y + size), 
                                    self.LINE_WIDTH)
                    
                elif self.game.board[row][col] == 'O':
                    size = min(cell_width, cell_height) // 2 - 15
                    offset = 5
                    
                    pygame.draw.circle(self.screen, self.DARK_GRAY, 
                                    (center_x + offset, center_y + offset), 
                                    size, self.LINE_WIDTH)
                    
                    pygame.draw.circle(self.screen, self.BLUE, 
                                    (center_x, center_y), 
                                    size, self.LINE_WIDTH)
        
        # Draw animations
        for anim in self.animations:
            anim.draw(self.screen, self.LINE_WIDTH, self.RED, self.BLUE, self.DARK_GRAY)
        
        # Draw win line animation
        if self.win_animation:
            self.win_animation.update()
            self.win_animation.draw(self.screen, self.LINE_WIDTH)

    def handle_cell_click(self, mouse_pos):
        x, y = mouse_pos
        # Adjust for board position and size
        board_x = x - 100
        board_y = y - 200
        
        # Check if click is within board bounds
        if 0 <= board_x <= (self.WIDTH-200) and 0 <= board_y <= (self.HEIGHT-350):
            col = board_x // ((self.WIDTH-200) // self.BOARD_COLS)
            row = board_y // ((self.HEIGHT-350) // self.BOARD_ROWS)
            
            if 0 <= row < 3 and 0 <= col < 3:
                if self.game.make_move(row, col):                    
                    # Create animation
                    cell_width = (self.WIDTH-200) // self.BOARD_COLS
                    cell_height = (self.HEIGHT-350) // self.BOARD_ROWS
                    center_x = col * cell_width + 100 + cell_width // 2
                    center_y = row * cell_height + 200 + cell_height // 2
                    
                    size = min(cell_width, cell_height) // 2 - 15
                    self.animations.append(
                        MarkAnimation(
                            self.game.board[row][col],  # Current mark type
                            center_x,
                            center_y,
                            size
                        )
                    )
                    return row, col
        return None

    def display_winner(self):
        if self.game.winner:
            # Create semi-transparent overlay
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            # Winner announcement with background
            result_text = f"{self.game.winner} Wins!" if self.game.winner != "Draw" else "Draw!"
            color = self.RED if self.game.winner == 'X' else self.BLUE if self.game.winner == 'O' else self.BLACK
            
            # Text background
            text_surf = self.title_font.render(result_text, True, color)
            bg_rect = text_surf.get_rect(center=(self.WIDTH//2, self.HEIGHT//2))
            bg_rect.inflate_ip(40, 40)
            pygame.draw.rect(self.screen, self.WHITE, bg_rect, 0, border_radius=15)
            pygame.draw.rect(self.screen, color, bg_rect, 3, border_radius=15)
            
            # Text
            self.screen.blit(text_surf, text_surf.get_rect(center=(self.WIDTH//2, self.HEIGHT//2)))

    def check_win_line(self):
        if self.game.winner and self.game.winner != "Draw" and not self.win_animation:
            cell_width = (self.WIDTH-200) // self.BOARD_COLS
            cell_height = (self.HEIGHT-350) // self.BOARD_ROWS
            
            # Check rows
            for row in range(3):
                if (self.game.board[row][0] == self.game.board[row][1] == 
                    self.game.board[row][2] == self.game.winner):
                    start_x = 110
                    start_y = row * cell_height + 200 + cell_height // 2
                    end_x = self.WIDTH - 110
                    end_y = start_y
                    color = self.RED if self.game.winner == 'X' else self.BLUE
                    self.win_animation = WinLineAnimation((start_x, start_y), (end_x, end_y), color)
                    return
            
            # Check columns
            for col in range(3):
                if (self.game.board[0][col] == self.game.board[1][col] == 
                    self.game.board[2][col] == self.game.winner):
                    start_x = col * cell_width + 100 + cell_width // 2
                    start_y = 210
                    end_x = start_x
                    end_y = self.HEIGHT - 140
                    color = self.RED if self.game.winner == 'X' else self.BLUE
                    self.win_animation = WinLineAnimation((start_x, start_y), (end_x, end_y), color)
                    return
            
            # Check diagonals
            if self.game.board[0][0] == self.game.board[1][1] == self.game.board[2][2] == self.game.winner:
                start_x = 110
                start_y = 210
                end_x = self.WIDTH - 110
                end_y = self.HEIGHT - 140
                color = self.RED if self.game.winner == 'X' else self.BLUE
                self.win_animation = WinLineAnimation((start_x, start_y), (end_x, end_y), color)
                return
                
            if self.game.board[0][2] == self.game.board[1][1] == self.game.board[2][0] == self.game.winner:
                start_x = self.WIDTH - 110
                start_y = 210
                end_x = 110
                end_y = self.HEIGHT - 140
                color = self.RED if self.game.winner == 'X' else self.BLUE
                self.win_animation = WinLineAnimation((start_x, start_y), (end_x, end_y), color)
                return
            
    def reset_game(self):
        self.game = TicTacToe()  # Create a new game instance instead of just resetting
        self.animations = []
        self.win_animation = None
        self.score_updated = False
        
        # If player chose 'O', let AI make the first move
        if self.game_mode == "AI" and self.player_marker == 'O':
            pygame.time.delay(500)  # Brief delay for UX
            
            try:
                if self.selected_ai == "Minimax":
                    ai_row, ai_col = self.minimax_ai.best_move(self.game.board, depth_limit=5)
                elif self.selected_ai == "Q-Learning":
                    ai_row, ai_col = self.q_learning_ai.best_move(self.game.board)
                elif self.selected_ai == "DQN" and self.dqn_ai:
                    ai_row, ai_col = self.dqn_ai.best_move(self.game.board)
                elif self.selected_ai == "MCTS" and self.mcts_ai:
                    self.mcts_ai.set_player('X' if self.player_marker == 'O' else 'O')
                    ai_row, ai_col = self.mcts_ai.best_move(self.game.board, time_limit=2.0)
                else:
                    valid_moves = self.game.get_available_moves()
                    ai_row, ai_col = random.choice(valid_moves) if valid_moves else (0, 0)
            except Exception as e:
                print(f"AI error: {e}. Using random fallback.")
                valid_moves = self.game.get_available_moves()
                ai_row, ai_col = random.choice(valid_moves) if valid_moves else (0, 0)
                
            # Make AI's move
            if self.game.make_move(ai_row, ai_col):
                # Create animation for AI move
                cell_width = (self.WIDTH-200) // self.BOARD_COLS
                cell_height = (self.HEIGHT-350) // self.BOARD_ROWS
                center_x = ai_col * cell_width + 100 + cell_width // 2
                center_y = ai_row * cell_height + 200 + cell_height // 2
                    
                size = min(cell_width, cell_height) // 2 - 15
                self.animations.append(
                    MarkAnimation(
                        self.game.board[ai_row][ai_col],
                        center_x,
                        center_y,
                        size
                    )
                )

    def run_game(self):
        if not self.main_menu():
            return
            
        running = True
        
        while running:
            # Update animations and win status
            if self.game.winner and not self.score_updated:
                self.update_scoreboard()
            
            self.check_win_line()  # Check for win animations
            
            # Draw everything
            new_game_rect, undo_rect, redo_rect = self.draw_board()
            self.draw_marks()
            self.draw_scoreboard()
            if self.game.winner:
                self.display_winner()
            
            pygame.display.update()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    
                    if new_game_rect.collidepoint(mouse_pos):
                        self.reset_game()
                    elif undo_rect.collidepoint(mouse_pos):
                        if self.game.undo():
                            self.animations = []  # Clear animations on undo
                            self.win_animation = None
                            self.score_updated = False
                    elif redo_rect.collidepoint(mouse_pos):
                        if self.game.redo():
                            self.animations = []  # Clear animations on redo
                            self.win_animation = None
                            self.score_updated = False
                    elif (self.game.winner is None and 
                        (self.game_mode == "Human" or 
                        (self.game_mode == "AI" and 
                            ((self.player_marker == 'X' and self.game.current_player == 'X') or
                            (self.player_marker == 'O' and self.game.current_player == 'O'))))):
                        # Human's turn
                        cell = self.handle_cell_click(mouse_pos)
                        if cell:  # If a valid move was made
                            # If AI mode and game not over, make AI move
                            if self.game_mode == "AI" and self.game.winner is None:
                                pygame.time.delay(200)
                                
                            try:
                                if self.selected_ai == "Minimax":
                                    ai_row, ai_col = self.minimax_ai.best_move(self.game.board, depth_limit=4)
                                elif self.selected_ai == "Q-Learning":
                                    ai_row, ai_col = self.q_learning_ai.best_move(self.game.board)
                                elif self.selected_ai == "DQN" and self.dqn_ai:
                                    ai_row, ai_col = self.dqn_ai.best_move(self.game.board)
                                elif self.selected_ai == "MCTS" and self.mcts_ai:
                                    # For MCTS, set the player and pass time limit
                                    self.mcts_ai.set_player('X' if self.player_marker == 'O' else 'O')
                                    ai_row, ai_col = self.mcts_ai.best_move(self.game.board, time_limit=1.0)
                                else:  # Default to random if AI is not properly initialized
                                    valid_moves = self.game.get_available_moves()
                                    ai_row, ai_col = random.choice(valid_moves) if valid_moves else (0, 0)
                            except Exception as e:
                                print(f"AI error: {e}. Using random fallback.")
                                valid_moves = self.game.get_available_moves()
                                ai_row, ai_col = random.choice(valid_moves) if valid_moves else (0, 0)
                                
                            if self.game.make_move(ai_row, ai_col):
                                # Create animation for AI move
                                cell_width = (self.WIDTH-200) // self.BOARD_COLS
                                cell_height = (self.HEIGHT-350) // self.BOARD_ROWS
                                center_x = ai_col * cell_width + 100 + cell_width // 2
                                center_y = ai_row * cell_height + 200 + cell_height // 2
                                    
                                size = min(cell_width, cell_height) // 2 - 15
                                self.animations.append(
                                    MarkAnimation(
                                        self.game.board[ai_row][ai_col],
                                        center_x,
                                        center_y,
                                        size
                                    )
                                )
        pygame.quit()
    

class MarkAnimation:
    """Animation for X and O markers appearing on the board"""
    def __init__(self, mark_type, center_x, center_y, final_size):
        self.mark_type = mark_type  # 'X' or 'O'
        self.center_x = center_x
        self.center_y = center_y
        self.final_size = final_size
        self.current_size = 0
        self.growth_speed = final_size / 8  # Takes 8 frames to reach full size
        self.complete = False
        
    def update(self):
        if self.current_size < self.final_size:
            self.current_size += self.growth_speed
        else:
            self.current_size = self.final_size
            self.complete = True
            
    def draw(self, screen, line_width, red_color, blue_color, dark_gray):
        size = int(self.current_size)
        if size <= 0:
            return
            
        if self.mark_type == 'X':
            # Draw shadow
            pygame.draw.line(screen, dark_gray, 
                          (self.center_x - size + 5, self.center_y - size + 5), 
                          (self.center_x + size + 5, self.center_y + size + 5), 
                          line_width)
            pygame.draw.line(screen, dark_gray,
                             (self.center_x + size + 5, self.center_y - size + 5),
                             (self.center_x - size + 5, self.center_y + size + 5),  
                             line_width)
            
            # Draw X
            pygame.draw.line(screen, red_color, 
                          (self.center_x - size, self.center_y - size), 
                          (self.center_x + size, self.center_y + size), 
                          line_width)
            pygame.draw.line(screen, red_color, 
                          (self.center_x + size, self.center_y - size), 
                          (self.center_x - size, self.center_y + size), 
                          line_width)
        else:  # 'O'
            # Draw shadow
            pygame.draw.circle(screen, dark_gray, 
                            (self.center_x + 5, self.center_y + 5), 
                            size, line_width)
            
            # Draw O
            pygame.draw.circle(screen, blue_color, 
                            (self.center_x, self.center_y), 
                            size, line_width)


class WinLineAnimation:
    """Animation for the winning line"""
    def __init__(self, start_pos, end_pos, color):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.color = color
        self.current_length = 0
        self.total_length = ((end_pos[0] - start_pos[0])**2 + 
                            (end_pos[1] - start_pos[1])**2)**0.5
        self.growth_speed = self.total_length / 15  # Takes 15 frames to complete
        self.complete = False
        
    def update(self):
        if self.current_length < self.total_length:
            self.current_length += self.growth_speed
        else:
            self.current_length = self.total_length
            self.complete = True
            
    def draw(self, screen, line_width):
        progress = min(1.0, self.current_length / self.total_length)
        current_x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * progress
        current_y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * progress
        
        pygame.draw.line(screen, self.color, 
                        self.start_pos, 
                        (current_x, current_y), 
                        line_width)

if __name__ == "__main__":
    gui = TicTacToeGUI()
    gui.run_game()