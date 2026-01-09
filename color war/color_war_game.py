import pygame
import sys
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 5
CELL_SIZE = 80
MARGIN = 10
WINDOW_SIZE = GRID_SIZE * CELL_SIZE + 2 * MARGIN
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (220, 50, 50)
BLUE = (50, 100, 220)
LIGHT_RED = (255, 150, 150)
LIGHT_BLUE = (150, 180, 255)
DARK_RED = (150, 20, 20)
DARK_BLUE = (20, 50, 150)


class Cell:
    def __init__(self):
        self.atoms = 0
        self.color = 0  # 0: neutral, 1: red, -1: blue


class ChainReactionGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 100))
        pygame.display.set_caption("Chain Reaction Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self.grid = [[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.current_player = 1  # 1: red, -1: blue
        self.first_round = True
        self.first_move_red = True
        self.first_move_blue = True
        self.red_count = 0
        self.blue_count = 0
        self.game_over = False
        self.winner = None
        self.animating = False
        self.animation_queue = deque()

    def is_valid_move(self, x, y):
        """Check if a move is valid"""
        if self.game_over:
            return False

        cell = self.grid[x][y]

        if self.first_round:
            # First round: can only place on empty cells
            return cell.color == 0
        else:
            # After first round: can ONLY place on own colored cells
            return cell.color == self.current_player

    def add_atom(self, x, y):
        """Add an atom to a cell"""
        if not self.is_valid_move(x, y):
            return False

        cell = self.grid[x][y]

        # First move for each player adds 3 atoms
        if (self.current_player == 1 and self.first_move_red) or \
                (self.current_player == -1 and self.first_move_blue):
            cell.atoms += 3
            if self.current_player == 1:
                self.first_move_red = False
            else:
                self.first_move_blue = False
        else:
            cell.atoms += 1

        cell.color = self.current_player

        # Update player counts
        if self.current_player == 1:
            self.red_count += 1
        else:
            self.blue_count += 1

        # Check for explosions
        self.check_explosions()

        # Check for game over after explosions
        if not self.first_round:
            self.check_game_over()

        if not self.game_over:
            # Switch player
            self.current_player *= -1

            # After both players have moved once, first round is over
            if self.current_player == 1 and self.red_count > 0 and self.blue_count > 0:
                self.first_round = False

        return True

    def check_explosions(self):
        """Check and handle explosions recursively"""
        exploded = True
        while exploded:
            exploded = False
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    cell = self.grid[x][y]

                    # Explosion happens at 4 atoms regardless of position
                    if cell.atoms >= 4:
                        exploded = True
                        self.explode(x, y)

            # After each explosion wave, update counts
            self.update_cell_counts()

    def explode(self, x, y):
        """Explode a cell and spread to neighbors"""
        cell = self.grid[x][y]
        color = cell.color
        cell.atoms = 0
        cell.color = 0

        # Spread to neighbors
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                neighbor = self.grid[nx][ny]
                neighbor.atoms += 1
                neighbor.color = color

    def update_cell_counts(self):
        """Update red and blue cell counts"""
        self.red_count = 0
        self.blue_count = 0

        for row in self.grid:
            for cell in row:
                if cell.color == 1:
                    self.red_count += 1
                elif cell.color == -1:
                    self.blue_count += 1

    def check_game_over(self):
        """Check if the game is over"""
        if self.red_count == 0:
            self.game_over = True
            self.winner = "Blue"
        elif self.blue_count == 0:
            self.game_over = True
            self.winner = "Red"

    def draw_atoms(self, x, y, atoms, color):
        """Draw atoms in a cell"""
        cell_x = MARGIN + y * CELL_SIZE
        cell_y = MARGIN + x * CELL_SIZE
        center_x = cell_x + CELL_SIZE // 2
        center_y = cell_y + CELL_SIZE // 2

        atom_color = RED if color == 1 else BLUE
        radius = 12

        if atoms == 1:
            pygame.draw.circle(self.screen, atom_color, (center_x, center_y), radius)
        elif atoms == 2:
            pygame.draw.circle(self.screen, atom_color, (center_x - 15, center_y), radius)
            pygame.draw.circle(self.screen, atom_color, (center_x + 15, center_y), radius)
        elif atoms == 3:
            pygame.draw.circle(self.screen, atom_color, (center_x, center_y - 15), radius)
            pygame.draw.circle(self.screen, atom_color, (center_x - 15, center_y + 10), radius)
            pygame.draw.circle(self.screen, atom_color, (center_x + 15, center_y + 10), radius)
        elif atoms >= 4:
            pygame.draw.circle(self.screen, atom_color, (center_x - 15, center_y - 15), radius)
            pygame.draw.circle(self.screen, atom_color, (center_x + 15, center_y - 15), radius)
            pygame.draw.circle(self.screen, atom_color, (center_x - 15, center_y + 15), radius)
            pygame.draw.circle(self.screen, atom_color, (center_x + 15, center_y + 15), radius)

    def draw(self):
        """Draw the game board"""
        self.screen.fill(WHITE)

        # Draw grid
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell = self.grid[i][j]
                x = MARGIN + j * CELL_SIZE
                y = MARGIN + i * CELL_SIZE

                # Draw cell background
                if cell.color == 1:
                    bg_color = LIGHT_RED
                elif cell.color == -1:
                    bg_color = LIGHT_BLUE
                else:
                    bg_color = GRAY

                pygame.draw.rect(self.screen, bg_color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 2)

                # Draw atoms
                if cell.atoms > 0:
                    self.draw_atoms(i, j, cell.atoms, cell.color)

        # Draw player info
        info_y = WINDOW_SIZE + 20

        if self.game_over:
            text = self.font.render(f"{self.winner} Wins!", True, BLACK)
            self.screen.blit(text, (WINDOW_SIZE // 2 - text.get_width() // 2, info_y))

            restart_text = self.small_font.render("Press R to Restart", True, BLACK)
            self.screen.blit(restart_text, (WINDOW_SIZE // 2 - restart_text.get_width() // 2, info_y + 40))
        else:
            player_text = "Red's Turn" if self.current_player == 1 else "Blue's Turn"
            player_color = RED if self.current_player == 1 else BLUE
            text = self.font.render(player_text, True, player_color)
            self.screen.blit(text, (WINDOW_SIZE // 2 - text.get_width() // 2, info_y))

            # Draw counts
            count_text = self.small_font.render(f"Red: {self.red_count}  Blue: {self.blue_count}", True, BLACK)
            self.screen.blit(count_text, (WINDOW_SIZE // 2 - count_text.get_width() // 2, info_y + 40))

        pygame.display.flip()

    def handle_click(self, pos):
        """Handle mouse click"""
        x, y = pos

        # Check if click is within grid
        if MARGIN <= x < WINDOW_SIZE - MARGIN and MARGIN <= y < WINDOW_SIZE - MARGIN:
            col = (x - MARGIN) // CELL_SIZE
            row = (y - MARGIN) // CELL_SIZE

            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                self.add_atom(row, col)

    def reset(self):
        """Reset the game"""
        self.grid = [[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.current_player = 1
        self.first_round = True
        self.first_move_red = True
        self.first_move_blue = True
        self.red_count = 0
        self.blue_count = 0
        self.game_over = False
        self.winner = None

    def run(self):
        """Main game loop"""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset()

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


# Run the game
if __name__ == "__main__":
    game = ChainReactionGame()
    game.run()